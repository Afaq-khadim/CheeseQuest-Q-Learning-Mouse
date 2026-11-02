# ============================================================
# FYP: Reinforcement Learning — Multi-Agent Maze
# Competition + Cooperation + +--Dynamic Maze
# Agents: 2 Q-Learning + 1 A* (for comparison)
# ============================================================

import pygame
import numpy as np
import random
import heapq

# ── CONFIGURATION ────────────────────────────────────────────
WIDTH, HEIGHT   = 1100, 680
ROWS,  COLS     = 10, 10
CELL_SIZE       = 540 // COLS
FPS             = 10

# ── COLORS ───────────────────────────────────────────────────
WHITE       = (255, 255, 255)
BLACK       = (0,   0,   0  )
BLUE        = (30,  144, 255)
LIGHT_BLUE  = (173, 216, 230)
RED         = (220, 50,  50 )
LIGHT_RED   = (255, 180, 180)
GREEN       = (0,   200, 0  )
LIGHT_GREEN = (180, 255, 180)
YELLOW      = (255, 220, 0  )
GREY        = (200, 200, 200)
DARK_GREY   = (50,  50,  50 )
ORANGE      = (255, 165, 0  )
PURPLE      = (180, 0,   255)
BG_COLOR    = (25,  25,  35 )
PANEL_BG    = (40,  40,  55 )

# ── Q-LEARNING PARAMETERS ────────────────────────────────────
ALPHA           = 0.5
GAMMA           = 0.95
EPSILON         = 1.0
EPSILON_MIN     = 0.01
EPSILON_DECAY   = 0.7
GUIDED_EPISODES = 15
MAX_MOVES       = 300

ACTIONS      = [(-1,0),(1,0),(0,-1),(0,1)]
ACTION_NAMES = ["Up","Down","Left","Right"]

# ── SHARED Q-TABLE (Cooperation) ─────────────────────────────
# Both Q-Learning agents share knowledge
shared_q_table = np.zeros((ROWS, COLS, 4))

# ── MAZE FUNCTIONS ───────────────────────────────────────────
def generate_maze(rows, cols, start, goal, wall_prob=0.25):
    while True:
        grid = [[0]*cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if (r,c) in [start, goal]:
                    continue
                if random.random() < wall_prob:
                    grid[r][c] = 1
        if has_path(grid, start, goal):
            return grid

def has_path(grid, start, goal):
    visited = set()
    queue   = [start]
    while queue:
        curr = queue.pop(0)
        if curr == goal:
            return True
        if curr in visited:
            continue
        visited.add(curr)
        r, c = curr
        for dr, dc in ACTIONS:
            nr, nc = r+dr, c+dc
            if 0<=nr<ROWS and 0<=nc<COLS and grid[nr][nc]==0:
                queue.append((nr,nc))
    return False

def solve_astar(grid, start, goal):
    def h(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = []
    heapq.heappush(open_set, (h(start,goal), 0, start))
    came_from = {}
    g_score   = {start: 0}
    while open_set:
        _, g, curr = heapq.heappop(open_set)
        if curr == goal:
            break
        r, c = curr
        for dr, dc in ACTIONS:
            nr, nc   = r+dr, c+dc
            neighbor = (nr, nc)
            if 0<=nr<ROWS and 0<=nc<COLS and grid[nr][nc]==0:
                tentative = g + 1
                if neighbor not in g_score or tentative < g_score[neighbor]:
                    g_score[neighbor] = tentative
                    f = tentative + h(neighbor, goal)
                    heapq.heappush(open_set, (f, tentative, neighbor))
                    came_from[neighbor] = curr
    path = []
    curr = goal
    while curr != start:
        if curr not in came_from:
            return []
        path.append(curr)
        curr = came_from[curr]
    path.append(start)
    path.reverse()
    return path

# ── Q-LEARNING AGENT ─────────────────────────────────────────
class QLearningAgent:
    def __init__(self, agent_id, start, goal, color, light_color, name):
        self.agent_id       = agent_id
        self.start          = start
        self.goal           = goal
        self.pos            = start
        self.color          = color
        self.light_color    = light_color
        self.name           = name
        self.epsilon        = EPSILON
        self.score          = 0
        self.moves          = 0
        self.episode        = 0
        self.episode_reward = 0
        self.reached        = False
        self.best_moves     = float('inf')
        self.wins           = 0
        self.total_rewards  = []

    def reset(self):
        self.pos            = self.start
        self.moves          = 0
        self.episode_reward = 0
        self.reached        = False

    def choose_action(self, grid, other_positions):
        r,  c  = self.pos
        gr, gc = self.goal

        # First 15 episodes — guide toward goal
        if self.episode < GUIDED_EPISODES:
            best_action = None
            best_dist   = float('inf')
            for i, (dr, dc) in enumerate(ACTIONS):
                nr, nc = r+dr, c+dc
                if (0<=nr<ROWS and 0<=nc<COLS and
                        grid[nr][nc]==0 and
                        (nr,nc) not in other_positions):
                    dist = abs(nr-gr) + abs(nc-gc)
                    if dist < best_dist:
                        best_dist   = dist
                        best_action = i
            if best_action is not None:
                return best_action

        # Epsilon-greedy using shared Q-Table
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(shared_q_table[r, c])

    def update_q_table(self, state, action, reward, next_state):
        r,  c  = state
        nr, nc = next_state
        old_val    = shared_q_table[r, c, action]
        future_max = np.max(shared_q_table[nr, nc])
        new_val    = old_val + ALPHA * (
            reward + GAMMA * future_max - old_val
        )
        shared_q_table[r, c, action] = new_val

    def step(self, grid, other_positions):
        if self.reached:
            return 0

        action   = self.choose_action(grid, other_positions)
        dr, dc   = ACTIONS[action]
        r,  c    = self.pos
        nr, nc   = r+dr, c+dc
        prev_pos = self.pos

        # Collision with other agents = penalty
        if (nr, nc) in other_positions:
            reward = -15
            nr, nc = r, c

        elif 0<=nr<ROWS and 0<=nc<COLS and grid[nr][nc]==0:
            self.pos   = (nr, nc)
            self.moves += 1

            if self.pos == self.goal:
                reward       = 200 + max(0, 100 - self.moves)
                self.reached = True
                self.score  += 10
                self.wins   += 1
                if self.moves < self.best_moves:
                    self.best_moves = self.moves
                print(f"🏆 {self.name} reached goal in episode "
                      f"{self.episode} | Moves: {self.moves}")
            else:
                old_dist = abs(r  - self.goal[0]) + abs(c  - self.goal[1])
                new_dist = abs(nr - self.goal[0]) + abs(nc - self.goal[1])
                reward   = 2 if new_dist < old_dist else -2
        else:
            reward = -20
            nr, nc = r, c

        self.update_q_table(prev_pos, action, reward, (nr,nc))
        self.episode_reward += reward
        return reward

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.episode += 1
        self.total_rewards.append(self.episode_reward)

# ── A* AGENT ─────────────────────────────────────────────────
class AStarAgent:
    def __init__(self, start, goal):
        self.start      = start
        self.goal       = goal
        self.pos        = start
        self.color      = GREEN
        self.light_color= LIGHT_GREEN
        self.name       = "A* Agent"
        self.score      = 0
        self.moves      = 0
        self.wins       = 0
        self.reached    = False
        self.path       = []
        self.step_idx   = 0
        self.best_moves = float('inf')

    def compute_path(self, grid):
        self.path     = solve_astar(grid, self.pos, self.goal)
        self.step_idx = 0

    def reset(self, grid):
        self.pos     = self.start
        self.moves   = 0
        self.reached = False
        self.compute_path(grid)

    def step(self, grid):
        if self.reached or self.step_idx >= len(self.path):
            return
        self.pos = self.path[self.step_idx]
        self.step_idx += 1
        self.moves    += 1
        if self.pos == self.goal:
            self.reached  = True
            self.score   += 10
            self.wins    += 1
            if self.moves < self.best_moves:
                self.best_moves = self.moves
            print(f"🟢 A* Agent reached goal | Moves: {self.moves}")

# ── DRAW MAZE ────────────────────────────────────────────────
def draw_maze(screen, grid, goal, agents, font):
    for r in range(ROWS):
        for c in range(COLS):
            x    = c * CELL_SIZE + 20
            y    = r * CELL_SIZE + 20
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, DARK_GREY if grid[r][c]==1 else WHITE, rect)
            pygame.draw.rect(screen, GREY, rect, 1)

    # Goal
    gr, gc = goal
    gx = gc * CELL_SIZE + 20
    gy = gr * CELL_SIZE + 20
    pygame.draw.rect(screen, YELLOW, (gx+2, gy+2, CELL_SIZE-4, CELL_SIZE-4))
    g_lbl = font.render("G", True, BLACK)
    screen.blit(g_lbl, (gx+8, gy+6))

    # Draw each agent
    for agent in agents:
        ar, ac = agent.pos
        ax = ac * CELL_SIZE + 20
        ay = ar * CELL_SIZE + 20
        # Glow
        pygame.draw.rect(screen, agent.light_color,
                         (ax-2, ay-2, CELL_SIZE+4, CELL_SIZE+4))
        # Body
        pygame.draw.rect(screen, agent.color,
                         (ax+2, ay+2, CELL_SIZE-4, CELL_SIZE-4))
        lbl = font.render(agent.name[0], True, WHITE)
        screen.blit(lbl, (ax+7, ay+5))

# ── DRAW PANEL ───────────────────────────────────────────────
def draw_panel(screen, agents, episode, font, big_font):
    px = COLS * CELL_SIZE + 45

    # Title
    title = big_font.render("Multi-Agent RL", True, YELLOW)
    screen.blit(title, (px, 15))
    sub = font.render("Competition + Cooperation", True, ORANGE)
    screen.blit(sub, (px, 45))

    # Divider
    pygame.draw.line(screen, GREY, (px, 70), (px+350, 70), 1)

    # Leaderboard header
    y = 80
    hdr = big_font.render("🏆 Leaderboard", True, WHITE)
    screen.blit(hdr, (px, y))
    y += 35

    # Sort by score
    sorted_agents = sorted(agents, key=lambda a: a.score, reverse=True)
    rank_colors   = [YELLOW, GREY, ORANGE]

    for i, agent in enumerate(sorted_agents):
        rank  = ["🥇", "🥈", "🥉"][i]
        color = rank_colors[i]

        # Agent name & rank
        name_text = big_font.render(f"{rank} {agent.name}", True, agent.color)
        screen.blit(name_text, (px, y))
        y += 28

        # Stats
        stats = [
            f"  Score : {agent.score}",
            f"  Wins  : {agent.wins}",
            f"  Moves : {agent.moves}",
            f"  Best  : {agent.best_moves if agent.best_moves != float('inf') else 'N/A'}",
        ]
        for stat in stats:
            s = font.render(stat, True, color)
            screen.blit(s, (px, y))
            y += 22
        y += 8

    # Divider
    pygame.draw.line(screen, GREY, (px, y), (px+350, y), 1)
    y += 10

    # Q-Learning info
    ql_agents = [a for a in agents if isinstance(a, QLearningAgent)]
    if ql_agents:
        ep_text = font.render(f"Episode : {ql_agents[0].episode}", True, WHITE)
        screen.blit(ep_text, (px, y))
        y += 28

        # Epsilon bar
        eps = ql_agents[0].epsilon
        eps_text = font.render(f"Epsilon : {eps:.3f}", True, GREY)
        screen.blit(eps_text, (px, y))
        y += 22
        bar_w    = 250
        bar_fill = int(bar_w * eps)
        pygame.draw.rect(screen, DARK_GREY, (px, y, bar_w, 12))
        pygame.draw.rect(screen, ORANGE,    (px, y, bar_fill, 12))
        y += 25

        # Mode
        if ql_agents[0].episode < GUIDED_EPISODES:
            mode = font.render("Mode: 🧭 GUIDED", True, LIGHT_BLUE)
        else:
            mode = font.render("Mode: 🧠 Q-LEARNING", True, GREEN)
        screen.blit(mode, (px, y))
        y += 35

    # Divider
    pygame.draw.line(screen, GREY, (px, y), (px+350, y), 1)
    y += 10

    # Controls
    ctrl_title = font.render("Controls:", True, GREY)
    screen.blit(ctrl_title, (px, y))
    y += 25
    controls = [
        "SPACE = Fast / Slow",
        "R     = Restart",
        "Q     = Quit",
    ]
    for line in controls:
        text = font.render(line, True, DARK_GREY)
        screen.blit(text, (px, y))
        y += 24

# ── MAIN ─────────────────────────────────────────────────────
def main():
    global shared_q_table
    pygame.init()
    screen   = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("FYP: Multi-Agent RL Maze")
    clock    = pygame.time.Clock()
    font     = pygame.font.SysFont("Arial", 17)
    big_font = pygame.font.SysFont("Arial", 20, bold=True)

    start_pos = (0, 0)
    goal_pos  = (ROWS-1, COLS-1)

    def init_agents(grid):
        global shared_q_table
        shared_q_table = np.zeros((ROWS, COLS, 4))

        # Different start positions for each agent
        starts = [
            (0, 0),
            (0, COLS-1),
            (ROWS-1, 0),
        ]
        # Make sure starts are not walls
        valid_starts = []
        for s in starts:
            if grid[s[0]][s[1]] == 0:
                valid_starts.append(s)
            else:
                valid_starts.append(start_pos)

        a1 = QLearningAgent(1, valid_starts[0], goal_pos,
                            BLUE,  LIGHT_BLUE,  "QL-Blue")
        a2 = QLearningAgent(2, valid_starts[1], goal_pos,
                            RED,   LIGHT_RED,   "QL-Red")
        a3 = AStarAgent(valid_starts[2], goal_pos)
        a3.compute_path(grid)
        return [a1, a2, a3]

    grid    = generate_maze(ROWS, COLS, start_pos, goal_pos)
    agents  = init_agents(grid)
    speed   = 1
    episode = 0
    running = True

    while running:
        screen.fill(BG_COLOR)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    grid   = generate_maze(ROWS, COLS, start_pos, goal_pos)
                    agents = init_agents(grid)
                    episode = 0
                    print("🔄 Restarted!")
                if event.key == pygame.K_SPACE:
                    speed = 20 if speed == 1 else 1
                    print(f"⚡ Speed: {'FAST' if speed==20 else 'SLOW'}")

        for _ in range(speed):
            # Get all agent positions for collision detection
            other_positions = {a.pos for a in agents}

            for agent in agents:
                others = other_positions - {agent.pos}
                if isinstance(agent, QLearningAgent):
                    agent.step(grid, others)
                else:
                    agent.step(grid)

            # Add delay in slow mode
            if speed == 1:
                pygame.time.delay(120)

            # Check if all agents done or max moves reached
            ql_agents = [a for a in agents
                         if isinstance(a, QLearningAgent)]
            all_done  = all(
                a.reached or a.moves >= MAX_MOVES for a in agents
            )

            if all_done:
                episode += 1
                # Check winner (first to reach goal)
                winners = [a for a in agents if a.reached]
                if winners:
                    first = min(winners, key=lambda a: a.moves)
                    first.score += 5   # Bonus for winning the race!
                    print(f"🏁 Episode {episode} Winner: {first.name}")

                # Reset all agents
                for agent in agents:
                    if isinstance(agent, QLearningAgent):
                        agent.decay_epsilon()
                        agent.reset()
                    else:
                        agent.reset(grid)

                # Dynamic maze every 10 episodes
                if episode % 10 == 0:
                    grid = generate_maze(ROWS, COLS,
                                        start_pos, goal_pos)
                    for agent in agents:
                        if isinstance(agent, AStarAgent):
                            agent.compute_path(grid)
                    print(f"🌀 New maze generated at episode {episode}!")

        # Draw
        draw_maze(screen, grid, goal_pos, agents, font)
        draw_panel(screen, agents, episode, font, big_font)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

    # Final summary
    print("\n" + "="*45)
    print("📊 FINAL RESULTS")
    print("="*45)
    sorted_agents = sorted(agents, key=lambda a: a.score, reverse=True)
    for i, a in enumerate(sorted_agents):
        print(f"{'🥇' if i==0 else '🥈' if i==1 else '🥉'} "
              f"{a.name} | Score: {a.score} | "
              f"Wins: {a.wins} | Best: {a.best_moves}")

if __name__ == "__main__":
    main()