from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class MyMG_Env(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=10*size*size
        )
        self.goals_done = 0
        self.size = size;
    
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Create a vertical splitting wall
        splitIdx = round(width/2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(top=(2, 2), size=(1,1))

        # Place a door in the wall
        doorIdx = round(height/2)
        self.put_obj(Door('yellow'), splitIdx, doorIdx)

        # Place a ball (i.e. the apple) inside a box (i.e. the fridge) and that in the env
        apple = Ball(color='red')
        fridge = Box('blue', contains=apple)
        self.put_obj(fridge, round(width*3/4), 1)

        # Place a yellow key on the left side
        # self.place_obj(
        #     obj=Key('yellow'),
        #     top=(0, 0),
        #     size=(splitIdx, height)
        # )

        self.put_obj(Goal(), round(width/4), height-2)

        self.mission = "open the door, then open the fridge, then take apple and then put apple on table."
    
    def step(self, action):
        self.step_count += 1
        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos  
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.actions.pickup: 
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
            if self.carrying.type == 'ball' and self.goals_done == 2:
                self.goals_done = 3
                reward = self._reward()

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None
            if (fwd_pos == [round(self.size/4), self.size-2]).all() and self.goals_done == 3:
                self.goals_done = 0
                done = True
                reward = self._reward()

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)
            if fwd_cell.type == 'door' and self.goals_done == 0:
                self.goals_done = 1
                reward = self._reward()
            if fwd_cell.type == 'box' and self.goals_done == 1:
                self.goals_done = 2
                reward = self._reward()


        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

class Paper_Env(MyMG_Env):
    def __init__(self):
        super().__init__(size=10)

register(
    id='MiniGrid-Paper_Env-v0',
    entry_point='gym_minigrid.envs.MyEnvs:Paper_Env'
)