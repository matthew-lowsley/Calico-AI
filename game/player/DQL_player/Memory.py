

from collections import deque
import os
import random
import csv
from game.constants import MAX_MEMORY, Transition, REPLAY_RECENT

# Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class Memory():

    def __init__(self):
        self.queue = deque(maxlen=MAX_MEMORY)
        self.save_to_csv = False
        self.save_limit = 20_000
    
    def push(self, *args):
        self.queue.append(Transition(*args))
        if self.save_to_csv and len(self.queue) >= self.save_limit:
            self.save()
            quit()
    
    def sample(self, batch_size):
        if batch_size > len(self.queue):
            return self.queue
        else:
            # Append most recent transitions 
            sample = random.sample(self.queue, batch_size-REPLAY_RECENT)
            sample.append(self.queue[-REPLAY_RECENT])
            #print("Length of Sample: "+str(len(sample)))
            #print("Sample Type: "+str(type(sample)))
            return sample

    def save(self):

        with open("transitions.csv", 'w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(['state', 'action', 'reward', 'next_state', 'done'])

            for transition in self.queue:
                writer.writerow([
                    transition.state.tolist(),
                    transition.action.tolist(),
                    transition.reward,
                    transition.next_state.tolist(),
                    transition.done
                ])