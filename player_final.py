from board_hex import BoardHex
from player_hex import PlayerHex
from seahorse.game.light_action import LightAction
from seahorse.game.game_state import GameState
from seahorse.game.game_layout.board import Piece
from game_state_hex import GameStateHex
from collections import deque
import math
import random
import heapq
import time
import copy

class MyPlayer(PlayerHex):
    """
    Hex AI Player using Minimax with Alpha-Beta pruning and advanced heuristics.
    Combines multiple strategic approaches for optimal play.
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer", *args) -> None:
        super().__init__(piece_type, name, *args)
        self.transposition_table = {}
        self.zobrist_keys = self._init_zobrist()
        self.killer_moves = {}
        self.dijkstra_cache = {}

    def compute_action(self, current_state: GameState, remaining_time: float = 60, **kwargs) -> LightAction:
        """
        Main decision function with adaptive depth based on remaining time.
        Uses iterative deepening and killer move heuristic.
        """
        board = current_state.get_rep()
        env = board.env
        dim = board.dimensions[0]

        legal_actions = list(current_state.generate_possible_light_actions())
        if not legal_actions:
            return None

        # Opening: play center on first move
        if len(env) == 0:
            center = dim // 2
            candidates = [
                (center, center), (center-1, center), (center, center-1), 
                (center+1, center), (center, center+1)
            ]
            # Filter valid positions
            valid_candidates = [pos for pos in candidates if 0 <= pos[0] < dim and 0 <= pos[1] < dim]
            return LightAction({"piece": self.piece_type, "position": valid_candidates[0]})

        # Second move: strategic response
        if len(env) == 1:
            opp_pos = list(env.keys())[0]
            center = (dim // 2, dim // 2)
            
            # Always take center if available
            if center not in env:
                return LightAction({"piece": self.piece_type, "position": center})
            
            # Otherwise play mirror move
            mirror_i = dim - 1 - opp_pos[0]
            mirror_j = dim - 1 - opp_pos[1]
            if (mirror_i, mirror_j) not in env:
                return LightAction({"piece": self.piece_type, "position": (mirror_i, mirror_j)})
            
            # Fallback: strong central position
            strong_backup = (dim // 2, dim // 2 - 1)  # Or other central cell
            return LightAction({"piece": self.piece_type, "position": strong_backup})
            
        # Check if we can win THIS move**
        winning_move = self._find_winning_move(current_state, legal_actions)
        if winning_move is not None:
            return winning_move
        
        opp_piece = self._opponent()
        opp_dist = self._shortest_path_dijkstra(board, opp_piece)

        # Check if opponent can win NEXT move and block it**
        blocking_move = self._find_blocking_move(current_state, legal_actions)
        if blocking_move is not None:
            return blocking_move

        # Aggressive defense with jumping when opponent is moderately close
        if opp_dist < 8:
            aggressive_block_move = self.aggressive_defense(board, self.piece_type, opp_piece)
            if aggressive_block_move is not None:
                return aggressive_block_move

        # Clear TT periodically to manage memory
        if len(self.transposition_table) > 100000:
            self.transposition_table.clear()
            self.killer_moves.clear()

        # Adaptive depth based on remaining time and board state
        empties = dim * dim - len(env)
        
        if empties > 180:  # Opening
            max_depth = 2
        elif empties > 120:  # Midgame
            max_depth = 3 if remaining_time > 15 else 2
        elif empties > 60:   # Endgame approaching
            max_depth = 4 if remaining_time > 20 else 3
        else:               # Critical endgame
            max_depth = 5 if remaining_time > 10 else 4

        # Move ordering
        legal_actions = self._order_actions_fast(legal_actions, board)
        candidates = legal_actions[:min(30, len(legal_actions))]  # Limit branching factor

        best_score = -math.inf
        best_action = candidates[0]
        start_time = time.time()
        budget_per_move = max(0.1, remaining_time / max(1, empties / 8))

        for idx, action in enumerate(candidates):
            elapsed = time.time() - start_time
            if elapsed > budget_per_move * 0.85:
                break

            next_state = current_state.apply_action(action)
            score = self._minimax(next_state, max_depth - 1, -math.inf, math.inf, False)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _find_winning_move(self, state: GameStateHex, actions: list) -> LightAction:
        """
        Check if any move leads to immediate win.
        """
        for action in actions:
            next_state = state.apply_action(action)
            if next_state.is_done():
                winner = self._check_winner(next_state)
                if winner == self.piece_type:
                    return action
        return None
    
    def _find_blocking_move(self, state: GameStateHex, actions: list) -> LightAction:
        """
        Check if opponent can win in one move, and block it if possible.
        """
        board = state.get_rep()
        opp_piece = self._opponent()
        opp_dist = self._shortest_path_dijkstra(board, opp_piece)
        
        # Only block if opponent is close to winning
        if opp_dist > 5.0:
            return None
        
        # Try blocking the opponent's shortest path
        for action in actions:
            next_state = state.apply_action(action)
            next_board = next_state.get_rep()
            new_opp_dist = self._shortest_path_dijkstra(next_board, opp_piece)
            
            # If blocking move increases opponent's distance significantly
            if new_opp_dist > opp_dist + 2:
                return action
        
        return None

    def aggressive_defense(self, board, my_piece: str, opp_piece: str) -> LightAction:
        """
        Strategic blocking by "jumping" on opponent's path.
        Combines blocking effectiveness with own progression.
        """
        env = board.env
        dim = board.dimensions[0]
        
        opp_path = self._reconstruct_shortest_path(board, opp_piece)
        
        if not opp_path or len(opp_path) < 3:
            return None
        
        # Calculate our path once for efficiency
        my_path = self._reconstruct_shortest_path(board, my_piece)
        
        blocking_candidates = []
        
        # Target central third of opponent's path
        start_idx = min(2, len(opp_path) // 3)
        end_idx = min(len(opp_path) - 1, 2 * len(opp_path) // 3)
        
        for i in range(start_idx, end_idx):
            pos = opp_path[i]
            
            for neighbor in self._get_neighbors(pos, dim):
                if neighbor not in env or not env[neighbor]:
                    # Check not directly adjacent to opponent stones
                    is_adjacent_to_opp = False
                    for adj in self._get_neighbors(neighbor, dim):
                        if adj in env and env[adj] and env[adj].get_type() == opp_piece:
                            is_adjacent_to_opp = True
                            break
                    
                    if not is_adjacent_to_opp:
                        block_score = self._evaluate_block_position(
                            neighbor, board, my_piece, opp_piece, opp_path, my_path
                        )
                        blocking_candidates.append((neighbor, block_score))
        
        if not blocking_candidates:
            return None
        
        blocking_candidates.sort(key=lambda x: x[1], reverse=True)
        best_pos = blocking_candidates[0][0]
        
        return LightAction({"piece": my_piece, "position": best_pos})

    def _reconstruct_shortest_path(self, board, piece: str) -> list:
        """
        Reconstruct the shortest path (list of positions).
        Returns the complete path from source to target.
        """
        dim = board.dimensions[0]
        env = board.env
        INF = 1e9
        
        dist = {}
        parent = {}
        pq = []
        
        if piece == "R":
            for j in range(dim):
                pos = (0, j)
                cell = env.get(pos)
                if cell is None:
                    dist[pos] = 1.0
                    heapq.heappush(pq, (1.0, pos))
                elif cell.get_type() == "R":
                    dist[pos] = 0.0
                    heapq.heappush(pq, (0.0, pos))
            goal_check = lambda p: p[0] == dim - 1
        else:
            for i in range(dim):
                pos = (i, 0)
                cell = env.get(pos)
                if cell is None:
                    dist[pos] = 1.0
                    heapq.heappush(pq, (1.0, pos))
                elif cell.get_type() == "B":
                    dist[pos] = 0.0
                    heapq.heappush(pq, (0.0, pos))
            goal_check = lambda p: p[1] == dim - 1
        
        goal_node = None
        while pq:
            d, pos = heapq.heappop(pq)
            if d > dist.get(pos, INF):
                continue
            if goal_check(pos):
                goal_node = pos
                break
            i, j = pos
            for _, (n_type, n_pos) in board.get_neighbours(i, j).items():
                if n_type == "OUTSIDE":
                    continue
                ni, nj = n_pos
                if n_type == piece:
                    nd = d
                elif n_type == "EMPTY":
                    nd = d + 1.0
                else:
                    continue
                if nd < dist.get(n_pos, INF):
                    dist[n_pos] = nd
                    parent[n_pos] = pos
                    heapq.heappush(pq, (nd, n_pos))
        
        if goal_node is None:
            return []
        
        path = []
        current = goal_node
        while current in parent:
            path.append(current)
            current = parent[current]
        path.append(current)
        
        return list(reversed(path))
    
    

    def _get_neighbors(self, pos: tuple, dim: int) -> list:
        """Get valid neighbors of a position."""
        i, j = pos
        candidates = [
            (i-1, j), (i-1, j+1),
            (i, j-1), (i, j+1),
            (i+1, j-1), (i+1, j)
        ]
        return [(ni, nj) for ni, nj in candidates if 0 <= ni < dim and 0 <= nj < dim]

    def _evaluate_block_position(self, pos: tuple, board, my_piece: str, 
                                opp_piece: str, opp_path: list, my_path: list = None) -> float:
        """
        Evaluate the quality of a blocking position.
        Considers multiple strategic factors.
        """
        dim = board.dimensions[0]
        env = board.env
        score = 0.0
        
        # Bonus for cutting opponent's path
        path_positions_cut = 0
        for neighbor in self._get_neighbors(pos, dim):
            if neighbor in opp_path:
                path_positions_cut += 1
        score += path_positions_cut * 50.0
        
        # Bonus for proximity to own pieces
        my_neighbors = 0
        for neighbor in self._get_neighbors(pos, dim):
            if neighbor in env and env[neighbor] and env[neighbor].get_type() == my_piece:
                my_neighbors += 1
        score += my_neighbors * 30.0
        
        # Bonus for creating bridges with our pieces
        for neighbor in self._get_neighbors(pos, dim):
            for second_neighbor in self._get_neighbors(neighbor, dim):
                if second_neighbor in env and env[second_neighbor]:
                    if env[second_neighbor].get_type() == my_piece:
                        score += 20.0
                        break
        
        # Penalty for being too close to edges
        distance_from_edge = min(pos[0], pos[1], dim - 1 - pos[0], dim - 1 - pos[1])
        if distance_from_edge < 2:
            score -= 25.0
        
        # Major bonus if on our own path (dual-purpose move)
        if my_path and pos in my_path:
            score += 100.0
        
        return score

 
    def _minimax(self, state: GameStateHex, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """
        Minimax with alpha-beta pruning, transposition table, and killer moves.
        """
        # Terminal state check
        if state.is_done():
            winner = self._check_winner(state)
            if winner == self.piece_type:
                return 1e7 + (4 - depth)  # Prefer faster wins
            elif winner is not None:
                return -1e7 - (4 - depth)
            return 0

        # Leaf node: evaluate heuristically
        if depth == 0:
            return self._evaluate(state)

        # Transposition table lookup
        zobrist = self._hash_state(state)
        maximizing_str = "max" if maximizing else "min"
        tt_key = f"{zobrist}_{depth}_{maximizing_str}"
        
        if tt_key in self.transposition_table:
            return self.transposition_table[tt_key]

        board = state.get_rep()
        actions = list(state.generate_possible_light_actions())

        if not actions:
            val = self._evaluate(state)
            self.transposition_table[tt_key] = val
            return val

        # Move ordering: killer moves first, then positional ordering
        actions = self._order_actions_fast(actions, board)
        actions = actions[:min(30, len(actions))]  # Limit branching factor

        if maximizing:
            value = -math.inf
            best_move = None
            
            for action in actions:
                next_state = state.apply_action(action)
                v = self._minimax(next_state, depth - 1, alpha, beta, False)
                
                if v > value:
                    value = v
                    best_move = action
                
                alpha = max(alpha, value)
                if alpha >= beta:
                    # Beta cutoff: record killer move
                    if depth not in self.killer_moves:
                        self.killer_moves[depth] = []
                    if best_move and len(self.killer_moves[depth]) < 2:
                        self.killer_moves[depth].append(best_move)
                    break
        else:
            value = math.inf
            best_move = None
            
            for action in actions:
                next_state = state.apply_action(action)
                v = self._minimax(next_state, depth - 1, alpha, beta, True)
                
                if v < value:
                    value = v
                    best_move = action
                
                beta = min(beta, value)
                if beta <= alpha:
                    # Alpha cutoff: record killer move
                    if depth not in self.killer_moves:
                        self.killer_moves[depth] = []
                    if best_move and len(self.killer_moves[depth]) < 2:
                        self.killer_moves[depth].append(best_move)
                    break

        self.transposition_table[tt_key] = value
        return value
    
    def _evaluate(self, state: GameStateHex) -> float:
        """
        Advanced heuristic evaluation function combining multiple features.
        """
        board = state.get_rep()
        my_piece = self.piece_type
        opp_piece = self._opponent()

        # Primary feature: shortest path distances (multi-source Dijkstra)
        my_dist = self._shortest_path_dijkstra(board, my_piece)
        opp_dist = self._shortest_path_dijkstra(board, opp_piece)

        # Immediate one-move win/loss detection (hard thresholds)
        if my_dist <= 1.0:
            return 1e6
        if opp_dist <= 1.0:
            return -1e6

        # Path score (dominant)
        path_score = (opp_dist - my_dist) * 10000.0  # strong weight

        # Side-aware connectivity: how well components are positioned for the goal
        my_conn = self._side_connectivity_score(board, my_piece)
        opp_conn = self._side_connectivity_score(board, opp_piece)
        conn_score = (my_conn - opp_conn) * 4.0

        # Virtual connections (bridges, 2-step potentials)
        my_virtual = self._count_virtual_connections(board, my_piece)
        opp_virtual = self._count_virtual_connections(board, opp_piece)
        virtual_score = (my_virtual - opp_virtual) * 3000.0

        # Blocking effectiveness: reward stones that lie on opponent's shortest paths
        block_score = self._blocking_effectiveness(board, my_piece, opp_piece) * 700.0

        # Cut vulnerability: penalize fragile opponent/own shortest path (we want robust paths)
        my_vul = self._cut_vulnerability(board, my_piece)
        opp_vul = self._cut_vulnerability(board, opp_piece)
        vuln_score = (opp_vul - my_vul) * 600.0

        total = path_score + conn_score + virtual_score + block_score + vuln_score

        # clamp to safe bounds
        return max(-1e6, min(1e6, total))

    
    def _shortest_path_dijkstra(self, board, piece: str) -> float:
            """
            Multi-source Dijkstra from all border source cells for the given piece.
            Costs: own stone = 0, empty = 1, opponent = blocked.
            Returns shortest cost to reach the opposite border. If unreachable, returns large constant.
            """
            cache_key = self._hash_board_for_dijkstra(board, piece)

            if cache_key in self.dijkstra_cache:
                return self.dijkstra_cache[cache_key]

            dim = board.dimensions[0]
            env = board.env
            INF = 1e9

            dist = {}
            pq = []

            if piece == "R":
                # sources: top row (i == 0)
                for j in range(dim):
                    pos = (0, j)
                    cell = env.get(pos)
                    if cell is None:
                        dist[pos] = 1.0
                        heapq.heappush(pq, (1.0, pos))
                    elif cell.get_type() == "R":
                        dist[pos] = 0.0
                        heapq.heappush(pq, (0.0, pos))
                goal_check = lambda p: p[0] == dim - 1
            else:
                # sources: left column (j == 0)
                for i in range(dim):
                    pos = (i, 0)
                    cell = env.get(pos)
                    if cell is None:
                        dist[pos] = 1.0
                        heapq.heappush(pq, (1.0, pos))
                    elif cell.get_type() == "B":
                        dist[pos] = 0.0
                        heapq.heappush(pq, (0.0, pos))
                goal_check = lambda p: p[1] == dim - 1

            while pq:
                d, pos = heapq.heappop(pq)
                if d > dist.get(pos, INF):
                    continue
                if goal_check(pos):
                    self.dijkstra_cache[cache_key] = d
                    return d
                i, j = pos
                for _, (n_type, n_pos) in board.get_neighbours(i, j).items():
                    if n_type == "OUTSIDE":
                        continue
                    ni, nj = n_pos
                    if n_type == piece:
                        nd = d
                    elif n_type == "EMPTY":
                        nd = d + 1.0
                    else:
                        # opponent stone: cannot pass
                        continue
                    if nd < dist.get(n_pos, INF):
                        dist[n_pos] = nd
                        heapq.heappush(pq, (nd, n_pos))

            # unreachable: return a large value
            result = dim * 3.0
            self.dijkstra_cache[cache_key] = result
            return result

    def _side_connectivity_score(self, board, piece: str) -> float:
        """
        Side-aware connectivity score:
        - Build union-find of same-color stones
        - Attach virtual nodes for each player's source/target sides
        - If virtual nodes are connected -> huge value
        - Otherwise compute a heuristic measure: how close the components get to
            connecting the two sides measured by minimal empty cells bridging them.
        Returns an integer-like score (higher is better).
        """
        dim = board.dimensions[0]
        env = board.env

        # map (i,j) -> index
        def idx(i, j):
            return i * dim + j

        n = dim * dim
        parent = list(range(n + 2))  # last two are virtual source/target
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra = find(a); rb = find(b)
            if ra != rb:
                parent[rb] = ra

        src = n
        tgt = n + 1

        # union same-color adjacent stones
        for (i, j), cell in env.items():
            if not cell or cell.get_type() != piece:
                continue
            a = idx(i, j)
            for _, (n_type, (ni, nj)) in board.get_neighbours(i, j).items():
                if n_type == piece:
                    union(a, idx(ni, nj))

        # attach border stones to virtual nodes depending on piece orientation
        if piece == "R":
            # top row -> src, bottom row -> tgt
            for j in range(dim):
                top = env.get((0, j))
                if top and top.get_type() == "R":
                    union(src, idx(0, j))
                bottom = env.get((dim - 1, j))
                if bottom and bottom.get_type() == "R":
                    union(tgt, idx(dim - 1, j))
        else:
            # left col -> src, right col -> tgt
            for i in range(dim):
                left = env.get((i, 0))
                if left and left.get_type() == piece:
                    union(src, idx(i, 0))
                right = env.get((i, dim - 1))
                if right and right.get_type() == piece:
                    union(tgt, idx(i, dim - 1))

        # if already connected, return very large value
        if find(src) == find(tgt):
            return 5000.0

        # otherwise compute minimal "bridge" distance between any node in src-component(s) and tgt-component(s)
        # BFS from all source-affiliated cells (treat own stones cost 0, empty cost 1)
        from collections import deque
        dist = {}
        dq = deque()

        if piece == "R":
            # initialize from top row cells that are either own or empty (favor own)
            for j in range(dim):
                p = (0, j)
                cell = env.get(p)
                if cell is None:
                    dist[p] = 1.0
                    dq.append(p)
                elif cell.get_type() == piece:
                    dist[p] = 0.0
                    dq.append(p)
            goal_check = lambda pos: pos[0] == dim - 1
        else:
            for i in range(dim):
                p = (i, 0)
                cell = env.get(p)
                if cell is None:
                    dist[p] = 1.0
                    dq.append(p)
                elif cell.get_type() == piece:
                    dist[p] = 0.0
                    dq.append(p)
            goal_check = lambda pos: pos[1] == dim - 1

        best = 1e9
        while dq:
            p = dq.popleft()
            d = dist[p]
            if goal_check(p):
                best = min(best, d)
                continue
            i, j = p
            for _, (n_type, npos) in board.get_neighbours(i, j).items():
                if n_type == "OUTSIDE":
                    continue
                if npos in dist:
                    continue
                cell = env.get(npos)
                if cell is None:
                    nd = d + 1.0
                elif cell.get_type() == piece:
                    nd = d
                else:
                    continue
                dist[npos] = nd
                dq.append(npos)

        if best > 1e8:
            return 0.0
        # invert distance to a positive connectivity measure
        # smaller best => bigger score
        return max(0.0, (30.0 - best) * 60.0)


    def _count_virtual_connections(self, board, piece: str) -> float:
        """
        Count common virtual connections (bridges and strong two-step patterns).
        This is a conservative detector: counts canonical bridges and simple 2-move potentials.
        """
        env = board.env
        dim = board.dimensions[0]
        checked = set()
        count = 0.0

        # helper: is inside and own piece
        def is_own(p):
            cell = env.get(p)
            return (cell is not None) and (cell.get_type() == piece)
        def is_empty(p):
            return env.get(p) is None

        # canonical hex bridge offsets (knight moves): two cells separated by a single shared neighbor
        knight_offsets = [(-2, 1), (-1, 2), (1, 1), (2, -1), (1, -2), (-1, -1)]
        # For each own stone, look for other own stone in knight offset -> check supporting empty cells
        for (i, j), cell in env.items():
            if not cell or cell.get_type() != piece:
                continue
            for dx, dy in knight_offsets:
                ni, nj = i + dx, j + dy
                if not (0 <= ni < dim and 0 <= nj < dim):
                    continue
                if not is_own((ni, nj)):
                    continue
                pair = tuple(sorted(((i, j), (ni, nj))))
                if pair in checked:
                    continue
                checked.add(pair)

                neigh1 = [npos for _, (_, npos) in board.get_neighbours(i, j).items() if board.get_neighbours(i, j)]
                neigh2 = [npos for _, (_, npos) in board.get_neighbours(ni, nj).items() if board.get_neighbours(ni, nj)]
                # count shared empties
                shared_empties = 0
                for p1 in neigh1:
                    if is_empty(p1):
                        for p2 in neigh2:
                            if p1 == p2:
                                shared_empties += 1
                # heuristic: if there's at least one shared empty neighbor, this is a potential bridge
                # if shared_empties >= 1:
                #     count += 1.0
                count += shared_empties
        return count


    def _blocking_effectiveness(self, board, my_piece: str, opp_piece: str) -> float:
        """
        Fast and reliable blocking evaluation that considers strategic zones,
        bridge disruption, and opponent progress blocking.
        """
        env = board.env
        dim = board.dimensions[0]
        blocking_score = 0.0
        
        # Get opponent's progress for context-aware blocking
        opp_dist = self._shortest_path_dijkstra(board, opp_piece)
        
        # Strategic blocking zones based on opponent's orientation
        if opp_piece == "R":
            # Opponent connects top-bottom: focus on central vertical corridor
            blocking_zones = []
            # Primary central columns
            for j in range(max(0, dim//2-2), min(dim, dim//2+3)):
                for i in range(dim):
                    blocking_zones.append((i, j))
            
            # Secondary: columns that are natural bridge points
            for j in [dim//4, 3*dim//4]:
                if 0 <= j < dim:
                    for i in range(dim):
                        blocking_zones.append((i, j))
        else:
            # Opponent connects left-right: focus on central horizontal corridor
            blocking_zones = []
            # Primary central rows
            for i in range(max(0, dim//2-2), min(dim, dim//2+3)):
                for j in range(dim):
                    blocking_zones.append((i, j))
            
            # Secondary: rows that are natural bridge points
            for i in [dim//4, 3*dim//4]:
                if 0 <= i < dim:
                    for j in range(dim):
                        blocking_zones.append((i, j))
        
        # Score our stones in blocking zones
        zone_positions = set(blocking_zones)
        for pos, cell in env.items():
            if cell and cell.get_type() == my_piece and pos in zone_positions:
                i, j = pos
                
                # Base score for being in blocking zone
                base_score = 1.0
                
                # Bonus for central positions (more critical blocking)
                center_bonus = 1.0 - (abs(i - dim//2) + abs(j - dim//2)) / (dim)
                base_score += center_bonus * 0.5
                
                # Bonus for disrupting opponent bridges
                bridge_disruption = self._count_disrupted_bridges(board, pos, opp_piece)
                base_score += bridge_disruption * 0.8
                
                # Bonus for being adjacent to opponent stones (active blocking)
                adjacent_opponents = 0
                for _, (_, npos) in board.get_neighbours(i, j).items():
                    ncell = env.get(npos)
                    if ncell and ncell.get_type() == opp_piece:
                        adjacent_opponents += 1
                base_score += adjacent_opponents * 0.3
                
                blocking_score += base_score
        
        # Extra bonus for stones that are in opponent's potential path
        # Use a simple heuristic based on opponent's orientation
        path_blocking = 0.0
        for pos, cell in env.items():
            if cell and cell.get_type() == my_piece:
                i, j = pos
                if opp_piece == "R":
                    # For vertical opponent, horizontal alignment matters less than vertical progress
                    if abs(j - dim//2) <= 3:  # In central columns
                        # Bonus based on how much this disrupts vertical flow
                        vertical_disruption = 1.0 - (abs(i - dim//2) / (dim//2))
                        path_blocking += vertical_disruption
                else:
                    # For horizontal opponent, vertical alignment matters less than horizontal progress
                    if abs(i - dim//2) <= 3:  # In central rows
                        horizontal_disruption = 1.0 - (abs(j - dim//2) / (dim//2))
                        path_blocking += horizontal_disruption
        
        blocking_score += path_blocking * 0.7
        
        # Scale based on how close opponent is to winning
        # More valuable blocking when opponent is close
        urgency_factor = max(0.5, 2.0 - (opp_dist / dim))
        blocking_score *= urgency_factor
        
        return blocking_score
    
    def _count_disrupted_bridges(self, board, pos: tuple, opp_piece: str) -> float:
        """
        Count how many opponent bridges this stone disrupts.
        """
        env = board.env
        dim = board.dimensions[0]
        disrupted = 0.0
        
        i, j = pos
        
        # Check if this stone breaks potential opponent bridges
        # Look for bridge patterns where this stone occupies a critical empty cell
        bridge_patterns = [
            # Horizontal bridge patterns
            [(-1, 0), (1, 1)], [(-1, -1), (1, 0)], [(-1, 1), (1, 0)],
            # Vertical bridge patterns
            [(0, -1), (1, 1)], [(0, 1), (1, -1)], [(-1, -1), (0, 1)]
        ]
        
        for pattern in bridge_patterns:
            # Check if this position could complete a bridge for opponent
            stones_found = 0
            empties_needed = 0
            
            for di, dj in pattern:
                ni, nj = i + di, j + dj
                if 0 <= ni < dim and 0 <= nj < dim:
                    cell = env.get((ni, nj))
                    if cell and cell.get_type() == opp_piece:
                        stones_found += 1
                    elif cell is None:
                        empties_needed += 1
            
            # If this pattern would be a bridge except for our stone here
            if stones_found == 2 and empties_needed == 0:
                disrupted += 1.0
            elif stones_found == 1 and empties_needed == 1:
                disrupted += 0.5
        
        return disrupted

    def _cut_vulnerability(self, board, piece: str) -> float:
        """
        Estimate how fragile the current (approximate) shortest path for `piece` is:
        - Compute the set of nodes on one shortest route (using Dijkstra parent links)
        - For each node, count alternative neighbors that maintain reachability.
        - Fragile if many nodes have low alternative support.
        Returns higher values for more fragile (bad) situations.
        """
        env = board.env
        dim = board.dimensions[0]

        # run Dijkstra while tracking parents to reconstruct one shortest path
        dist = {}
        parent = {}
        pq = []

        if piece == "R":
            for j in range(dim):
                pos = (0, j)
                cell = env.get(pos)
                if cell is None:
                    dist[pos] = 1.0
                    heapq.heappush(pq, (1.0, pos))
                elif cell.get_type() == "R":
                    dist[pos] = 0.0
                    heapq.heappush(pq, (0.0, pos))
            goal_check = lambda p: p[0] == dim - 1
        else:
            for i in range(dim):
                pos = (i, 0)
                cell = env.get(pos)
                if cell is None:
                    dist[pos] = 1.0
                    heapq.heappush(pq, (1.0, pos))
                elif cell.get_type() == piece:
                    dist[pos] = 0.0
                    heapq.heappush(pq, (0.0, pos))
            goal_check = lambda p: p[1] == dim - 1

        found_goal = None
        while pq:
            d, pos = heapq.heappop(pq)
            if d > dist.get(pos, 1e9):
                continue
            if goal_check(pos):
                found_goal = pos
                break
            i, j = pos
            for _, (n_type, npos) in board.get_neighbours(i, j).items():
                if n_type == "OUTSIDE":
                    continue
                if n_type == piece:
                    nd = d
                elif n_type == "EMPTY":
                    nd = d + 1.0
                else:
                    continue
                if nd < dist.get(npos, 1e9):
                    dist[npos] = nd
                    parent[npos] = pos
                    heapq.heappush(pq, (nd, npos))

        if found_goal is None:
            return 0.0

        # reconstruct one shortest path backward
        path = []
        cur = found_goal
        while cur in parent:
            path.append(cur)
            cur = parent[cur]
        # include start if present
        path.append(cur)
        path = list(reversed(path))

        # For each node on the path, count alternative neighbors that keep us on possibly-short routes.
        fragility = 0.0
        for p in path:
            i, j = p
            support = 0
            for _, (n_type, npos) in board.get_neighbours(i, j).items():
                if n_type == "OUTSIDE":
                    continue
                # neighbor supports path if it is our piece or empty and its presence gives equal-or-better reachability
                cell = env.get(npos)
                if cell is None or (cell and cell.get_type() == piece):
                    support += 1
            # low support -> fragile
            if support <= 2:
                fragility += (3 - support)  # 1 or 2 -> positive frag score
        return fragility

    def _order_actions_fast(self, actions: list, board) -> list:
        dim = board.dimensions[0]
        env = board.env
        my_piece = self.piece_type
        opp_piece = self._opponent()
        opp_dist = self._shortest_path_dijkstra(board, opp_piece)

        old_dist = self._shortest_path_dijkstra(board, my_piece)

        def score_action(action: LightAction):
            pos = action.data["position"]
            i, j = pos
            score = 0

            # PRIORITY -1: True shortest-path improvement
            copy_env = copy.copy(env)
            copy_env[pos] = Piece(piece_type=my_piece, owner=self)
            test_board = BoardHex(env=copy_env, dim=board.get_dimensions())

            new_dist = self._shortest_path_dijkstra(test_board, my_piece)
            dist_delta = old_dist - new_dist
            score -= dist_delta * 10000  # enormous weight, ensures you take the win

            if opp_dist <= 6:
                # Check if this move blocks opponent
                new_opp_dist = self._shortest_path_dijkstra(test_board, opp_piece)
                if new_opp_dist > opp_dist + 2: 
                    score -= 8000  

            # PRIORITY 0: Progress toward goal (still useful early game)
            if my_piece == "R":
                goal_distance = (dim - 1) - i
            else:
                goal_distance = (dim - 1) - j
            score -= goal_distance * 200

            # PRIORITY 1: Adjacent to own stones
            dist_to_own = self._min_dist_to_piece_type(env, pos, my_piece)
            if dist_to_own == 1:
                score -= 5000
            elif dist_to_own == 2:
                score -= 2500
            elif dist_to_own <= 3:
                score -= 500

            # PRIORITY 2: Blocking opponent threats
            dist_to_opp = self._min_dist_to_piece_type(env, pos, opp_piece)
            if dist_to_opp == 1:
                score -= 5000

            # random tie-break
            score += random.random() * 0.01

            return score

        return sorted(actions, key=score_action)


    def _min_dist_to_piece_type(self, env: dict, pos: tuple, piece_type: str) -> float:
        """Minimum distance to any piece of specific type."""
        pieces = [p for p, c in env.items() if c and c.get_type() == piece_type]
        if not pieces:
            return float('inf')
        return min(abs(pos[0] - p[0]) + abs(pos[1] - p[1]) for p in pieces)

    def _hash_state(self, state: GameStateHex) -> str:
        """Zobrist hashing for fast state evaluation."""
        h = 0
        env = state.get_rep().env
        for pos, cell in env.items():
            if cell:
                piece_idx = 0 if cell.get_type() == "R" else 1
                idx = pos[0] * 14 + pos[1]
                h ^= self.zobrist_keys[idx][piece_idx]
        return str(h)
    
    def _hash_board_for_dijkstra(self, board, piece: str) -> str:
        """
        Fast hashing specifically for Dijkstra caching.
        Only considers stones relevant to the given piece's pathfinding.
        """
        env = board.env
        
        # For efficiency, only hash positions that matter for this piece's connectivity
        relevant_positions = []
        
        for pos, cell in env.items():
            if cell:
                # Always include both colors since opponent stones block our path
                relevant_positions.append((pos, cell.get_type()))
        
        # Sort for consistent hashing
        relevant_positions.sort()
        
        # Include piece type in hash since paths are computed per piece
        hash_str = f"{piece}_{','.join(f'{pos[0]},{pos[1]},{ptype}' for pos, ptype in relevant_positions)}"
        
        return hash_str

    def _init_zobrist(self) -> list:
        """Initialize Zobrist random numbers (deterministic)."""
        random.seed(42)
        return [[(random.getrandbits(64)) for _ in range(2)] for _ in range(196)]

    def _check_winner(self, state: GameStateHex) -> str:
        """Check if game is won."""
        scores = state.get_scores()
        for player in state.players:
            if scores.get(player.get_id(), 0) == 1.0:
                return player.get_piece_type()
        return None

    def _opponent(self) -> str:
        """Return opponent piece type."""
        return "B" if self.piece_type == "R" else "R"