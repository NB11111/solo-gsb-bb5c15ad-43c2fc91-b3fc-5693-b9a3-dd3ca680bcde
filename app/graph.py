"""Branch script graph: validation (cycles / orphans / uncovered exits) and traversal."""
from __future__ import annotations

from collections import deque
from typing import Any


# ---- validation -----------------------------------------------------------

def validate(content: dict[str, Any]) -> dict[str, list]:
    """Return validation markers for a script.

    Markers:
      cycles  : ids of nodes that lie on / lead into a directed cycle
      orphans : ids of nodes unreachable from the start node
      uncovered: node ids that have no options and no explicit terminal flag,
                 or an option whose target is missing/empty
    Missing-target options are additionally reported under `errors`.
    """
    nodes: list[dict] = content.get("nodes") or []
    start_id = content.get("start_node_id")
    node_ids = [n.get("id") for n in nodes]
    node_map = {n.get("id"): n for n in nodes}

    adjacency: dict[Any, list[Any]] = {nid: [] for nid in node_ids}
    dangling_options: list[dict] = []  # {"node_id", "option_id"}

    for node in nodes:
        for opt in node.get("options", []) or []:
            target = opt.get("target_id")
            if target and target in node_map:
                adjacency[node["id"]].append(target)
            else:
                dangling_options.append({"node_id": node["id"], "option_id": opt.get("id")})

    cycles = _find_cycle_nodes(node_ids, adjacency)

    orphans: list = []
    if start_id in node_map:
        reachable = _bfs_reachable(start_id, adjacency)
        orphans = [nid for nid in node_ids if nid not in reachable]
    else:
        # No valid start: every node is effectively orphaned.
        orphans = list(node_ids)

    uncovered = [d["node_id"] for d in dangling_options]
    errors = [
        {
            "type": "missing_target",
            "node_id": d["node_id"],
            "option_id": d["option_id"],
            "message": "选项未连接到任何节点",
        }
        for d in dangling_options
    ]
    if nodes and start_id not in node_map:
        errors.append({"type": "missing_start", "message": "起始节点不存在"})

    for node in nodes:
        options = node.get("options", []) or []
        if not options and not node.get("terminal"):
            uncovered.append(node["id"])

    return {
        "cycles": sorted(set(cycles)),
        "orphans": sorted(set(orphans)),
        "uncovered": sorted(set(uncovered)),
        "errors": errors,
        "ok": not cycles and not orphans and not errors and not uncovered,
    }


def _bfs_reachable(start: Any, adjacency: dict[Any, list]) -> set:
    seen = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in adjacency.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _find_cycle_nodes(node_ids: list, adjacency: dict[Any, list]) -> list:
    """Nodes that belong to a directed cycle (Tarjan SCC, size>1 or self-loop)."""
    index_counter = [0]
    stack: list = []
    on_stack: set = set()
    indices: dict = {}
    lowlink: dict = {}
    cycle_nodes: set = set()

    def strongconnect(v):
        indices[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in adjacency.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            if len(component) > 1:
                cycle_nodes.update(component)
            elif v in adjacency.get(v, []):
                cycle_nodes.add(v)

    for nid in node_ids:
        if nid not in indices:
            strongconnect(nid)
    return list(cycle_nodes)


# ---- specified-option route planning --------------------------------------

def plan_route(content: dict[str, Any], required: list[dict[str, str]]) -> dict[str, Any]:
    """Shortest choice sequence covering every required option and reaching an end.

    Finds a path from ``start_node_id`` that:
      * chooses every required ``(node_id, option_id)`` at least once,
      * finishes at a terminal node — either explicitly marked terminal or a
        node without any options,
      * minimises the number of choices (BFS; cycles and repeated visits are
        allowed, so the search state is ``(node_id, covered-options-bitmask)``).

    Options of an explicitly terminal node can never be chosen. Structural
    problems (missing start / option, dangling target, terminal source,
    duplicates, count outside 1..6) raise ValueError; an impossible requirement
    set returns ``feasible: False`` without any partial path.
    """
    nodes: list[dict] = content.get("nodes") or []
    node_map = {n.get("id"): n for n in nodes}
    start_id = content.get("start_node_id")
    if start_id not in node_map:
        raise ValueError("起始节点不存在或未设置，无法规划路线")

    if not required:
        raise ValueError("请至少勾选一个选项")
    if len(required) > 6:
        raise ValueError("最多勾选 6 个选项")

    pairs: list[tuple[Any, Any]] = []
    seen_pairs: set[tuple[Any, Any]] = set()
    for item in required:
        nid, oid = item.get("node_id"), item.get("option_id")
        pair = (nid, oid)
        if pair in seen_pairs:
            raise ValueError("勾选的选项存在重复，请去重后再规划")
        seen_pairs.add(pair)
        node = node_map.get(nid)
        if node is None:
            raise ValueError(f"勾选选项所在的节点不存在：{nid}")
        opt = next((o for o in (node.get("options") or []) if o.get("id") == oid), None)
        if opt is None:
            raise ValueError(f"节点「{node.get('title') or nid}」上不存在勾选的选项")
        label = opt.get("label") or oid
        if node.get("terminal"):
            raise ValueError(f"选项「{label}」位于终止节点，终止节点不可继续选择，无法纳入规划")
        target = opt.get("target_id")
        if not target or target not in node_map:
            raise ValueError(f"选项「{label}」未连接到目标节点，只能勾选已连接目标的选项")
        pairs.append(pair)

    bit_of = {p: i for i, p in enumerate(pairs)}
    full_mask = (1 << len(pairs)) - 1

    def is_end(nid) -> bool:
        n = node_map[nid]
        return bool(n.get("terminal")) or not (n.get("options") or [])

    def successors(nid):
        # A terminal node is absorbing: its options can never be chosen.
        n = node_map[nid]
        if n.get("terminal"):
            return []
        out = []
        for o in n.get("options") or []:
            t = o.get("target_id")
            if t and t in node_map:
                out.append((o.get("id"), t))
        return out

    # BFS over (node_id, mask); every transition costs exactly one choice,
    # so the first goal state reached gives the minimum choice count.
    start_state = (start_id, 0)
    parent: dict[tuple, tuple | None] = {start_state: None}
    queue = deque([start_state])
    goal = None
    while queue:
        u, mask = queue.popleft()
        if mask == full_mask and is_end(u):
            goal = (u, mask)
            break
        for oid, target in successors(u):
            bit = bit_of.get((u, oid))
            next_mask = mask | (1 << bit) if bit is not None else mask
            next_state = (target, next_mask)
            if next_state not in parent:
                parent[next_state] = (u, mask, oid)
                queue.append(next_state)

    if goal is None:
        # Diagnostic: required options whose node is unreachable from the start
        # (following the same successor rules, i.e. terminal nodes absorb).
        adjacency = {nid: [t for _, t in successors(nid)] for nid in node_map}
        reachable = _bfs_reachable(start_id, adjacency)
        unreachable = []
        for nid, oid in pairs:
            if nid not in reachable:
                n = node_map[nid]
                o = next(x for x in (n.get("options") or []) if x.get("id") == oid)
                unreachable.append({
                    "node_id": nid, "option_id": oid,
                    "node_title": n.get("title", ""), "label": o.get("label", ""),
                })
        return {
            "feasible": False, "choice_count": None, "node_count": None,
            "steps": [], "end_node_id": None, "unreachable_required": unreachable,
            "reason": "从起始节点出发，无法在经过全部勾选选项后到达终止节点（含无选项节点）；"
                      "即使允许环路与重复访问节点，也不存在可行路线。",
        }

    # Reconstruct the chosen options from parent pointers.
    actions = []  # (from_node_id, option_id, target_node_id)
    cur = goal
    while parent[cur] is not None:
        prev_node, prev_mask, oid = parent[cur]
        actions.append((prev_node, oid, cur[0]))
        cur = (prev_node, prev_mask)
    actions.reverse()

    def brief(nid):
        n = node_map[nid]
        return {"id": nid, "title": n.get("title", ""),
                "type": n.get("type", "question"), "terminal": bool(n.get("terminal"))}

    steps = []
    cur_id = start_id
    for i, (from_id, oid, target_id) in enumerate(actions):
        n = node_map[from_id]
        o = next(x for x in (n.get("options") or []) if x.get("id") == oid)
        steps.append({
            "seq": i + 1,
            "node": brief(from_id),
            "choice": {"option_id": oid, "label": o.get("label", ""),
                       "required": (from_id, oid) in bit_of},
            "target": brief(target_id),
        })
        cur_id = target_id
    steps.append({"seq": len(actions) + 1, "node": brief(cur_id),
                  "choice": None, "target": None})

    return {
        "feasible": True, "choice_count": len(actions),
        "node_count": len(actions) + 1, "required_count": len(pairs),
        "end_node_id": cur_id, "steps": steps,
        "unreachable_required": [], "reason": None,
    }


# ---- runtime traversal ----------------------------------------------------

def step(content: dict, node_id: str, option_id: str | None = None) -> dict[str, Any]:
    """Resolve the next node from `node_id` via `option_id`.

    option_id None means "start": resolve from start_node_id.
    Returns {node, finished} or raises ValueError.
    """
    node_map = {n["id"]: n for n in content.get("nodes", [])}
    if option_id is None:
        start = content.get("start_node_id")
        if start not in node_map:
            raise ValueError("起始节点不存在")
        return {"node": node_map[start], "finished": False}

    current = node_map.get(node_id)
    if current is None:
        raise ValueError("当前节点不存在")
    opt = next((o for o in current.get("options", []) if o.get("id") == option_id), None)
    if opt is None:
        raise ValueError("该选项不存在")
    target = opt.get("target_id")
    if not target or target not in node_map:
        raise ValueError("选项未连接到目标节点")
    return {"node": node_map[target], "finished": False}
