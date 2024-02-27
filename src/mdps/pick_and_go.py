import numpy as np
import math
from gym import Env


class Point:

    def __init__(self, vec=None, x=None, y=None):
        if vec is not None:
            self.x = vec[0]
            self.y = vec[1]
        if x is not None and y is not None:
            self.x = x
            self.y = y

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Point(x=x, y=y)

    def __mul__(self, other):
        x = self.x * other
        y = self.y * other
        return Point(x=x, y=y)

    def __rmul__(self, other):
        x = self.x * other
        y = self.y * other
        return Point(x=x, y=y)

    def limit(self, minimum_point, maximum_point):
        self.x = min(self.x, maximum_point.x)
        self.x = max(self.x, minimum_point.x)
        self.y = min(self.y, maximum_point.y)
        self.y = max(self.y, minimum_point.y)

    def copy(self):
        return Point(x=self.x, y=self.y)

    def __str__(self):
        return f"Point({self.x}, {self.y})"

    def __repr__(self):
        return self.__str__()


class Square:

    def __init__(self, location=None, size=None):
        self.location = location
        self.size = size

    def __str__(self):
        return f"Square({self.location.__str__()}, {self.size})"


class Action:
    NOTHING = 0
    PICK = 1
    RELEASE = 2

    def __init__(self, acceleration, pick_action):
        self.acceleration = acceleration
        self.pick_action = pick_action  # 0 nothing. 1 pick. 2 release

    def __str__(self):
        return f"Action(acc={self.acceleration}, pick={self.pick_action})"


class Agent:
    PICK_STATE_NOTHING = 0
    PICK_STATE_HOLD = 1

    def __init__(self, location=Point([0, 0]), velocity=Point([0, 0]), pick_state=PICK_STATE_NOTHING):
        self.location = location
        self.velocity = velocity
        self.pick_state = 0  # 0 nothing. 1 hold.

    def __str__(self):
        return f"Agent(loc={self.location}, vel={self.velocity}, pick_state={self.pick_state})"


class PickAndGo(Env):
    """
    This environment is the simplest one containing both:
        1. continuous action (moving in the environment)
        2. discrete action (pick and release).
    The agent is a dot inside a rectangular in [0,1]x[0,1]. There is a reward
    when even the agent brings a box (size of 0.02 x 0.02) from a random place it is
    into the "target place" [0,0.02]x[0,0.02].
    When the agent (i.e., the dot) is inside a box, it can use the action "pick=1" to pick it up.
    As long as it is pick=1 the box moves with the agent.
    When pick=0 it releases the box.
    If the box is in the target-place then the agent occurs a reward = 1 and the episode ends.
    There is also some timeout to each episode.
    """

    def __init__(self,
                 seed=0,
                 target_size=0.02,
                 target_initial_location=Point([0, 0]),
                 box_size=0.02,
                 box_initial_location=Point([1.0, 1.0]),
                 world_size=1.0,
                 agent_initial_place=Point([0.5, 0.5]),
                 image_size=Point([100, 100]),
                 dt=0.1,  # sec
                 maximum_velocity=0.2,
                 maximum_acceleration=0.1,
                 ):
        super(PickAndGo, self).__init__()
        # States

        self.seed = seed
        self.target_size = target_size
        self.target_initial_location = target_initial_location
        self.box_size = box_size
        self.box_initial_location = box_initial_location
        self.world_size = world_size
        self.agent_initial_place = agent_initial_place
        self.image_size = image_size
        self.dt = dt

        # Inferred variables
        self.maximum_velocity = maximum_velocity
        self.maximum_acceleration = maximum_acceleration
        self.minimum_location = Point([0.0, 0.0])
        self.maximum_location = Point([self.world_size, self.world_size])

        # Define a 2-D observation space
        self.agent = None  # the first is x, the second is y
        self.box = None
        self.target = None
        self.random = np.random.RandomState(self.seed)

    @staticmethod
    def _cont2int(num, size):
        return min(math.floor(num * size), size - 1)

    def render(self):
        """
        The image is (y, x) where 0 is high in the y axis and more positive is lower in the image,
        and 0 is left in the x axis
        :return:

        RGB
        Red - box location
        Green - target-place
        Blue - agent location
        """

        image = np.zeros(shape=(3, self.image_size.y, self.image_size.x))
        # Agent
        agent_pixel_location_x = self._cont2int(self.agent.location.x, self.image_size.x)
        agent_pixel_location_y = self._cont2int(self.agent.location.y, self.image_size.y)
        image[2, agent_pixel_location_y, agent_pixel_location_x] = 1.0

        # Target place
        target_start_x = self._cont2int(self.target.location.x - self.target.size / 2, self.image_size.x)
        target_start_y = self._cont2int(self.target.location.y - self.target.size / 2, self.image_size.y)
        target_end_x = self._cont2int(self.target.location.x + self.target.size / 2, self.image_size.x)
        target_end_y = self._cont2int(self.target.location.y + self.target.size / 2, self.image_size.y)
        for x in range(target_start_x, target_end_x + 1):
            for y in range(target_start_y, target_end_y + 1):
                image[1, y, x] = 1.0

        # Agent place
        box_start_x = self._cont2int(self.box.location.x - self.box.size, self.image_size.x)
        box_start_y = self._cont2int(self.box.location.y - self.box.size, self.image_size.y)
        box_end_x = self._cont2int(self.box.location.x + self.box.size, self.image_size.x)
        box_end_y = self._cont2int(self.box.location.y + self.box.size, self.image_size.y)
        for x in range(box_start_x, box_end_x + 1):
            for y in range(box_start_y, box_end_y + 1):
                image[0, y, x] = 1.0
        return image

    def reset(self):
        # self.agent = Point(vec=self.random.uniform(0, 1, size=2))
        self.agent = Agent(location=self.agent_initial_place)
        self.box = Square(location=self.box_initial_location, size=self.box_size)
        self.target = Square(location=self.target_initial_location, size=self.target_size)
        image = self.render()
        return image

    def get_random_action(self):
        x_acc = self.random.uniform(0, self.maximum_acceleration)
        y_acc = self.random.uniform(0, self.maximum_acceleration)
        pick = self.random.choice(3, p=[0.9, 0.05, 0.05])
        return Action(acceleration=Point([x_acc, y_acc]), pick_action=pick)

    @staticmethod
    def point_on_square(point, square):
        return square.location.x - square.size / 2 <= point.x <= square.location.x + square.size / 2 and \
            square.location.y - square.size / 2 <= point.y <= square.location.y + square.size / 2

    def _get_info(self):
        info = dict()
        info["agent"] = self.agent.__str__()
        info["box"] = self.box.__str__()
        return info

    def step(self, action):
        self.agent.velocity = self.agent.velocity + action.acceleration * self.dt
        self.agent.location = self.agent.location + self.agent.velocity * self.dt
        self.agent.location.limit(self.minimum_location, self.maximum_location)
        print(f"{self.agent.velocity}  {self.agent.location}")

        if self.agent.pick_state == Agent.PICK_STATE_NOTHING and \
                action.pick_action == Action.PICK and \
                self.point_on_square(self.agent.location, self.box):
            self.agent.pick_state = Agent.PICK_STATE_HOLD
        elif self.agent.pick_state == Agent.PICK_STATE_HOLD and \
                action.pick_action == Action.RELEASE:
            self.agent.pick_state = Agent.PICK_STATE_NOTHING

        if self.agent.pick_state == Agent.PICK_STATE_HOLD:
            self.box.location = self.agent.location.copy()

        reward = PickAndGo.point_on_square(self.box.location, self.target)
        done = reward
        reward = int(reward)
        state_next = self.render()
        info = self._get_info()
        return state_next, reward, done, info

