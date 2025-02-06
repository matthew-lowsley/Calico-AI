import matplotlib.pyplot as plt

class Plotter():

    def __init__(self, n_players, ):

        plt.ion()

        self.ax = plt.subplot()
        self.n_players = n_players
        self.scores = [[] for _ in range(self.n_players)]
        self.lines = []

        for i in range(self.n_players):
            line, = self.ax.plot([], [], label=f"Player {i+1}")
            self.lines.append(line)

        self.ax.set_title("Agents Scores")
        self.ax.set_xlabel("Games")
        self.ax.set_ylabel("Mean Score")
        self.ax.legend()

    def plot(self, scores, n_games):

        for i in range(self.n_players):
            score = sum(scores[i]) / n_games
            self.scores[i].append(score)

        for i, line in enumerate(self.lines):
            line.set_ydata(self.scores[i])
            line.set_xdata(range(len(scores[i])))
        
        self.ax.set_title(f"Mean Agent Scores Over {n_games} Games")
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

        if n_games % 50 == 0:
            plt.savefig('DQL_Agent_Attempt1.png')
