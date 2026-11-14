"""Build execution-verified coding SFT data for the Qwen3.6-35B LoRA fine-tune.

Each example is a coding instruction plus a reference solution. We EXECUTE the
solution against its unit tests in an isolated subprocess and keep only the ones
that pass, so the training signal is verified-correct code (distillation, not
guesses). Output is JSONL with {prompt, completion}; the trainer renders the
prompt with the model chat template and masks it from the loss.

Solutions are stdlib-only and reinforce the patterns the eval rewards: clean
edge-case handling, list/tuple type consistency, correct algorithms.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parents[1]
# Verified data belongs to this suite, not to an engine checkout: Fennec is a
# native binary with no results/ tree, and Colibri's checkout must stay untouched.
OUT = Path(os.environ.get("FENNEC_RESULTS") or SUITE_ROOT / "results") / "sft_coding_verified.jsonl"

# (instruction, solution_code, test_code). test_code asserts against the solution.
BANK = [
    (
        "Write a function two_sum(nums, target) that returns the indices [i, j] of the two numbers adding to target, i < j. Assume exactly one solution. Use a hash map for O(n).",
        "def two_sum(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n        if target - x in seen:\n            return [seen[target - x], i]\n        seen[x] = i\n    return []\n",
        "assert two_sum([2,7,11,15], 9) == [0,1]\nassert two_sum([3,2,4], 6) == [1,2]\nassert two_sum([3,3], 6) == [0,1]\nassert two_sum([-1,-2,-3,-4], -6) == [1,3]\n",
    ),
    (
        "Write merge_intervals(intervals) returning a new list of merged [start, end] lists (not tuples), sorted by start. Handle empty input and touching intervals.",
        "def merge_intervals(intervals):\n    if not intervals:\n        return []\n    ordered = sorted([list(iv) for iv in intervals], key=lambda x: x[0])\n    out = [ordered[0]]\n    for s, e in ordered[1:]:\n        if s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out\n",
        "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\nassert merge_intervals([]) == []\nassert merge_intervals([[1,4],[4,5]]) == [[1,5]]\nassert merge_intervals([[1,4],[0,4]]) == [[0,4]]\nassert all(isinstance(x, list) for x in merge_intervals([[1,2]]))\n",
    ),
    (
        "Write parse_csv_line(line) that parses one CSV line into a list of string fields, honoring double quotes and escaped double-quotes (\"\"). An empty line returns ['']. Handle quoted fields containing commas.",
        "def parse_csv_line(line):\n    fields = []\n    cur = []\n    in_quotes = False\n    i = 0\n    n = len(line)\n    while i < n:\n        c = line[i]\n        if in_quotes:\n            if c == '\"':\n                if i + 1 < n and line[i+1] == '\"':\n                    cur.append('\"')\n                    i += 2\n                    continue\n                in_quotes = False\n                i += 1\n                continue\n            cur.append(c)\n            i += 1\n        else:\n            if c == '\"':\n                in_quotes = True\n                i += 1\n            elif c == ',':\n                fields.append(''.join(cur))\n                cur = []\n                i += 1\n            else:\n                cur.append(c)\n                i += 1\n    fields.append(''.join(cur))\n    return fields\n",
        "assert parse_csv_line('a,b,c') == ['a','b','c']\nassert parse_csv_line('') == ['']\nassert parse_csv_line('a,\"b,c\",d') == ['a','b,c','d']\nassert parse_csv_line('\"he said \"\"hi\"\"\",x') == ['he said \"hi\"','x']\nassert parse_csv_line('a,,b') == ['a','','b']\n",
    ),
    (
        "Write simplify_path(path) that resolves a Unix absolute path: collapse '//', resolve '.' and '..', remove trailing slashes. Root stays '/'.",
        "def simplify_path(path):\n    stack = []\n    for part in path.split('/'):\n        if part == '' or part == '.':\n            continue\n        if part == '..':\n            if stack:\n                stack.pop()\n        else:\n            stack.append(part)\n    return '/' + '/'.join(stack)\n",
        "assert simplify_path('/home/') == '/home'\nassert simplify_path('/../') == '/'\nassert simplify_path('/home//foo/') == '/home/foo'\nassert simplify_path('/a/./b/../../c/') == '/c'\nassert simplify_path('/') == '/'\n",
    ),
    (
        "Write length_of_lis(nums) returning the length of the longest strictly increasing subsequence in O(n log n) using patience sorting with bisect.",
        "import bisect\ndef length_of_lis(nums):\n    tails = []\n    for x in nums:\n        i = bisect.bisect_left(tails, x)\n        if i == len(tails):\n            tails.append(x)\n        else:\n            tails[i] = x\n    return len(tails)\n",
        "assert length_of_lis([10,9,2,5,3,7,101,18]) == 4\nassert length_of_lis([0,1,0,3,2,3]) == 4\nassert length_of_lis([7,7,7,7]) == 1\nassert length_of_lis([]) == 0\n",
    ),
    (
        "Write can_word_break(s, word_dict) returning True if s can be segmented into space-separated dictionary words. Use DP; words may be reused.",
        "def can_word_break(s, word_dict):\n    words = set(word_dict)\n    n = len(s)\n    dp = [False] * (n + 1)\n    dp[0] = True\n    for i in range(1, n + 1):\n        for j in range(i):\n            if dp[j] and s[j:i] in words:\n                dp[i] = True\n                break\n    return dp[n]\n",
        "assert can_word_break('leetcode', ['leet','code']) is True\nassert can_word_break('applepenapple', ['apple','pen']) is True\nassert can_word_break('catsandog', ['cats','dog','sand','and','cat']) is False\nassert can_word_break('', ['a']) is True\n",
    ),
    (
        "Write expr_eval(s) evaluating a non-negative integer expression with +, -, *, / (integer division truncating toward zero), parentheses and spaces. No eval(). Use a recursive descent or stack parser.",
        "def expr_eval(s):\n    tokens = []\n    i = 0\n    n = len(s)\n    while i < n:\n        c = s[i]\n        if c == ' ':\n            i += 1\n            continue\n        if c.isdigit():\n            j = i\n            while j < n and s[j].isdigit():\n                j += 1\n            tokens.append(int(s[i:j]))\n            i = j\n        else:\n            tokens.append(c)\n            i += 1\n    pos = 0\n    def peek():\n        return tokens[pos] if pos < len(tokens) else None\n    def parse_expr():\n        nonlocal pos\n        val = parse_term()\n        while peek() in ('+', '-'):\n            op = tokens[pos]; pos += 1\n            rhs = parse_term()\n            val = val + rhs if op == '+' else val - rhs\n        return val\n    def parse_term():\n        nonlocal pos\n        val = parse_factor()\n        while peek() in ('*', '/'):\n            op = tokens[pos]; pos += 1\n            rhs = parse_factor()\n            if op == '*':\n                val = val * rhs\n            else:\n                q = abs(val) // abs(rhs)\n                val = q if (val >= 0) == (rhs >= 0) else -q\n        return val\n    def parse_factor():\n        nonlocal pos\n        tok = tokens[pos]\n        if tok == '(':\n            pos += 1\n            val = parse_expr()\n            pos += 1\n            return val\n        if tok == '-':\n            pos += 1\n            return -parse_factor()\n        pos += 1\n        return tok\n    return parse_expr()\n",
        "assert expr_eval('3+2*2') == 7\nassert expr_eval(' 3/2 ') == 1\nassert expr_eval(' 3+5 / 2 ') == 5\nassert expr_eval('(1+(4+5+2)-3)+(6+8)') == 23\nassert expr_eval('2*(5+5*2)/3') == 10\nassert expr_eval('-3+5') == 2\n",
    ),
    (
        "Write rotate_matrix(matrix) rotating an n x n matrix 90 degrees clockwise IN PLACE. Transpose then reverse each row.",
        "def rotate_matrix(matrix):\n    n = len(matrix)\n    for i in range(n):\n        for j in range(i + 1, n):\n            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]\n    for row in matrix:\n        row.reverse()\n    return matrix\n",
        "assert rotate_matrix([[1,2,3],[4,5,6],[7,8,9]]) == [[7,4,1],[8,5,2],[9,6,3]]\nassert rotate_matrix([[1]]) == [[1]]\nassert rotate_matrix([[1,2],[3,4]]) == [[3,1],[4,2]]\n",
    ),
    (
        "Write topo_sort(n, edges) returning a lexicographically smallest topological order of nodes 0..n-1 given directed edges [u, v] (u before v), using a min-heap Kahn algorithm. Return [] if a cycle exists.",
        "import heapq\ndef topo_sort(n, edges):\n    adj = [[] for _ in range(n)]\n    indeg = [0] * n\n    for u, v in edges:\n        adj[u].append(v)\n        indeg[v] += 1\n    heap = [i for i in range(n) if indeg[i] == 0]\n    heapq.heapify(heap)\n    order = []\n    while heap:\n        u = heapq.heappop(heap)\n        order.append(u)\n        for v in adj[u]:\n            indeg[v] -= 1\n            if indeg[v] == 0:\n                heapq.heappush(heap, v)\n    return order if len(order) == n else []\n",
        "assert topo_sort(4, [[0,1],[0,2],[1,3],[2,3]]) == [0,1,2,3]\nassert topo_sort(2, [[0,1],[1,0]]) == []\nassert topo_sort(3, []) == [0,1,2]\nassert topo_sort(3, [[1,0],[2,0]]) == [1,2,0]\n",
    ),
    (
        "Write min_window_substring(s, t) returning the smallest substring of s containing every character of t with multiplicity, or '' if none. Use a sliding window with a need/have counter.",
        "from collections import Counter\ndef min_window_substring(s, t):\n    if not s or not t:\n        return ''\n    need = Counter(t)\n    missing = len(t)\n    best = (float('inf'), 0, 0)\n    left = 0\n    for right, ch in enumerate(s, 1):\n        if need[ch] > 0:\n            missing -= 1\n        need[ch] -= 1\n        if missing == 0:\n            while left < right and need[s[left]] < 0:\n                need[s[left]] += 1\n                left += 1\n            if right - left < best[0]:\n                best = (right - left, left, right)\n            need[s[left]] += 1\n            missing += 1\n            left += 1\n    return '' if best[0] == float('inf') else s[best[1]:best[2]]\n",
        "assert min_window_substring('ADOBECODEBANC', 'ABC') == 'BANC'\nassert min_window_substring('a', 'a') == 'a'\nassert min_window_substring('a', 'aa') == ''\nassert min_window_substring('', 'a') == ''\nassert min_window_substring('aa', 'aa') == 'aa'\n",
    ),
    (
        "Write an LRUCache class with get(key) and put(key, value), capacity-bounded, O(1), evicting the least recently used. Use OrderedDict or a dict + doubly linked list. get returns -1 if absent.",
        "from collections import OrderedDict\nclass LRUCache:\n    def __init__(self, capacity):\n        self.cap = capacity\n        self.d = OrderedDict()\n    def get(self, key):\n        if key not in self.d:\n            return -1\n        self.d.move_to_end(key)\n        return self.d[key]\n    def put(self, key, value):\n        if key in self.d:\n            self.d.move_to_end(key)\n        self.d[key] = value\n        if len(self.d) > self.cap:\n            self.d.popitem(last=False)\n",
        "c = LRUCache(2)\nc.put(1,1); c.put(2,2)\nassert c.get(1) == 1\nc.put(3,3)\nassert c.get(2) == -1\nc.put(4,4)\nassert c.get(1) == -1\nassert c.get(3) == 3\nassert c.get(4) == 4\n",
    ),
    (
        "Write rle_roundtrip(s) that run-length encodes s as counts+char (e.g. 'aaab' -> '3a1b') and a decoder; provide encode(s) and decode(s) such that decode(encode(s)) == s. Handle empty string.",
        "def encode(s):\n    if not s:\n        return ''\n    out = []\n    count = 1\n    for i in range(1, len(s)):\n        if s[i] == s[i-1]:\n            count += 1\n        else:\n            out.append(str(count) + s[i-1])\n            count = 1\n    out.append(str(count) + s[-1])\n    return ''.join(out)\ndef decode(s):\n    out = []\n    num = ''\n    for c in s:\n        if c.isdigit():\n            num += c\n        else:\n            out.append(c * int(num))\n            num = ''\n    return ''.join(out)\n",
        "assert encode('aaab') == '3a1b'\nassert decode('3a1b') == 'aaab'\nassert decode(encode('')) == ''\nassert decode(encode('aaaaaaaaaaab')) == 'aaaaaaaaaaab'\nassert encode('a') == '1a'\n",
    ),
    (
        "Write kth_smallest_sorted_matrix(matrix, k) returning the kth smallest element in an n x n matrix where each row and column is sorted ascending. Use a value-range binary search counting <= mid.",
        "def kth_smallest_sorted_matrix(matrix, k):\n    n = len(matrix)\n    lo, hi = matrix[0][0], matrix[-1][-1]\n    def count_le(x):\n        count = 0\n        row, col = n - 1, 0\n        while row >= 0 and col < n:\n            if matrix[row][col] <= x:\n                count += row + 1\n                col += 1\n            else:\n                row -= 1\n        return count\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if count_le(mid) < k:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo\n",
        "assert kth_smallest_sorted_matrix([[1,5,9],[10,11,13],[12,13,15]], 8) == 13\nassert kth_smallest_sorted_matrix([[-5]], 1) == -5\nassert kth_smallest_sorted_matrix([[1,2],[1,3]], 2) == 1\n",
    ),
    (
        "Write json_pointer(obj, pointer) resolving an RFC 6901 JSON pointer like '/foo/0/bar' against nested dicts/lists. '' returns obj. '~1' decodes to '/', '~0' to '~'. Raise KeyError/IndexError on missing.",
        "def json_pointer(obj, pointer):\n    if pointer == '':\n        return obj\n    if not pointer.startswith('/'):\n        raise ValueError('pointer must start with /')\n    cur = obj\n    for raw in pointer[1:].split('/'):\n        token = raw.replace('~1', '/').replace('~0', '~')\n        if isinstance(cur, list):\n            cur = cur[int(token)]\n        else:\n            cur = cur[token]\n    return cur\n",
        "doc = {'foo': ['bar', 'baz'], '': 0, 'a/b': 1, 'c~d': 2}\nassert json_pointer(doc, '') == doc\nassert json_pointer(doc, '/foo/0') == 'bar'\nassert json_pointer(doc, '/') == 0\nassert json_pointer(doc, '/a~1b') == 1\nassert json_pointer(doc, '/c~0d') == 2\n",
    ),
    (
        "Write atoi(s) parsing a leading optional sign and digits into a 32-bit signed integer, clamping to [-2**31, 2**31-1]. Ignore leading whitespace; stop at first non-digit; return 0 if no digits.",
        "def atoi(s):\n    INT_MIN, INT_MAX = -2**31, 2**31 - 1\n    i = 0\n    n = len(s)\n    while i < n and s[i] == ' ':\n        i += 1\n    sign = 1\n    if i < n and s[i] in '+-':\n        sign = -1 if s[i] == '-' else 1\n        i += 1\n    val = 0\n    while i < n and s[i].isdigit():\n        val = val * 10 + int(s[i])\n        i += 1\n    val *= sign\n    return max(INT_MIN, min(INT_MAX, val))\n",
        "assert atoi('42') == 42\nassert atoi('   -42') == -42\nassert atoi('4193 with words') == 4193\nassert atoi('words and 987') == 0\nassert atoi('-91283472332') == -2147483648\nassert atoi('') == 0\n",
    ),
    (
        "Write group_anagrams(strs) grouping words that are anagrams, returning a list of lists. Order of groups and within groups does not matter; compare by sorted membership.",
        "from collections import defaultdict\ndef group_anagrams(strs):\n    groups = defaultdict(list)\n    for w in strs:\n        groups[''.join(sorted(w))].append(w)\n    return list(groups.values())\n",
        "res = group_anagrams(['eat','tea','tan','ate','nat','bat'])\nflat = sorted(sorted(g) for g in res)\nassert flat == [['ate','eat','tea'],['bat'],['nat','tan']]\nassert group_anagrams(['']) == [['']]\nassert group_anagrams(['a']) == [['a']]\n",
    ),
    (
        "Write longest_palindrome(s) returning the longest palindromic substring using expand-around-center. Ties may return any longest; handle empty and single char.",
        "def longest_palindrome(s):\n    if not s:\n        return ''\n    start, end = 0, 0\n    def expand(l, r):\n        while l >= 0 and r < len(s) and s[l] == s[r]:\n            l -= 1\n            r += 1\n        return l + 1, r - 1\n    for i in range(len(s)):\n        l1, r1 = expand(i, i)\n        l2, r2 = expand(i, i + 1)\n        if r1 - l1 > end - start:\n            start, end = l1, r1\n        if r2 - l2 > end - start:\n            start, end = l2, r2\n    return s[start:end+1]\n",
        "assert longest_palindrome('babad') in ('bab', 'aba')\nassert longest_palindrome('cbbd') == 'bb'\nassert longest_palindrome('a') == 'a'\nassert longest_palindrome('') == ''\nassert longest_palindrome('racecar') == 'racecar'\n",
    ),
    (
        "Write coin_change(coins, amount) returning the fewest coins to make amount, or -1 if impossible. Use bottom-up DP; amount 0 needs 0 coins.",
        "def coin_change(coins, amount):\n    INF = amount + 1\n    dp = [0] + [INF] * amount\n    for a in range(1, amount + 1):\n        for c in coins:\n            if c <= a:\n                dp[a] = min(dp[a], dp[a - c] + 1)\n    return dp[amount] if dp[amount] != INF else -1\n",
        "assert coin_change([1,2,5], 11) == 3\nassert coin_change([2], 3) == -1\nassert coin_change([1], 0) == 0\nassert coin_change([1], 1) == 1\nassert coin_change([2,5,10,1], 27) == 4\n",
    ),
    (
        "Write is_valid_brackets(s) returning True if (), [], {} are balanced and properly nested. Use a stack; empty string is valid.",
        "def is_valid_brackets(s):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for c in s:\n        if c in pairs:\n            if not stack or stack.pop() != pairs[c]:\n                return False\n        elif c in '([{':\n            stack.append(c)\n    return not stack\n",
        "assert is_valid_brackets('()[]{}') is True\nassert is_valid_brackets('(]') is False\nassert is_valid_brackets('([)]') is False\nassert is_valid_brackets('{[]}') is True\nassert is_valid_brackets('') is True\n",
    ),
    (
        "Write flatten(nested) flattening an arbitrarily nested list of integers into a flat list, preserving order. Integers stay; do not flatten strings.",
        "def flatten(nested):\n    out = []\n    for item in nested:\n        if isinstance(item, list):\n            out.extend(flatten(item))\n        else:\n            out.append(item)\n    return out\n",
        "assert flatten([1,[2,[3,[4]],5]]) == [1,2,3,4,5]\nassert flatten([]) == []\nassert flatten([[[[]]]]) == []\nassert flatten([1,'ab',[2]]) == [1,'ab',2]\n",
    ),
]


def verify(code, tests):
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "t.py"
        f.write_text(code + "\n" + tests, encoding="utf-8")
        r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, timeout=30)
        return r.returncode == 0, r.stderr.strip()


def main():
    kept = 0
    rows = []
    for instr, code, tests in BANK:
        ok, err = verify(code, tests)
        if not ok:
            print(f"FAIL {instr[:50]!r}: {err}", flush=True)
            continue
        completion = "```python\n" + code.rstrip() + "\n```"
        rows.append({"prompt": instr, "completion": completion})
        kept += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"verified {kept}/{len(BANK)} examples -> {OUT}", flush=True)


if __name__ == "__main__":
    main()