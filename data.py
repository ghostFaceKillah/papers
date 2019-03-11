import attr


@attr.s
class Paper(object):
    title = attr.ib(type=str, default="")
    desc = attr.ib(type=str, default="")
    extra = attr.ib(type=str, default="")
    rating = attr.ib(type=int, default=0)
    priority = attr.ib(type=float, default=0.0)
    tags = attr.ib(type=set, factory=set)
    authors = attr.ib(type=list, factory=list)
    arxiv_id = attr.ib(type=str, default="")
    read = attr.ib(type=bool, default=False)
    url = attr.ib(type=str, default="")


DB = [
    Paper(
        desc="""
            We propose an expert-augmented actor-critic algorithm, which
            we evaluate on  two environments with sparse rewards: Montezumas Revenge
            and a demanding maze  from the ViZDoom suite. In the case of Montezumas
            Revenge, an agent trained  with our method achieves very good results
            consistently scoring above 27,000  points (in many experiments beating the
                first world). With an appropriate  choice of hyperparameters, our
            algorithm surpasses the performance of the  expert data. In a number of
            experiments, we have observed an unreported bug in  Montezumas Revenge
            which allowed the agent to score more than 800,000 points.
        """,
        title="Expert-augmented actor-critic for ViZDoom and Montezumas Revenge",
        extra='',
        tags=set('rl'), 
        authors=[
            u'Michal Garmulewicz', u'Henryk Michalewski', u'Piotr Milos'
        ],
        arxiv_id='1809.03447',
        read=True,
        rating=4
    ),

    Paper(
        title="DEEP REINFORCEMENT LEARNING WITH RELATIONAL INDUCTIVE BIASES",
        desc="""We introduce an approach for augmenting model-free deep reinforcement learning
                agents with a mechanism for relational reasoning over structured representations,
                which improves performance, learning efficiency, generalization, and interpretability.
                Our architecture encodes an image as a set of vectors, and applies an iterative
                message-passing procedure to discover and reason about relevant entities and relations in a scene.
                
                Bardzo grube to jest co oni robia. Korzystaja z multi head attention, aby wyciagnac
                z obrazka abstrakcyjna reprezentacje informacji, ktora pozniej przekazywana jest
                do agenta RL. Bardzo ciekawy wynik na box-world, oraz mocne wyniki (ale nie umiem
                do konca ocenic) na sub-taskach starcrafta.
                """,
        rating=9,
        url="https://openreview.net/pdf?id=HkxaFoC9KQ"
    ),
    Paper(
        title=""
    ),
    Paper(
        title='Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor',
        desc="""Model-free deep reinforcement learning (RL) algorithms have been demonstrated
        on a range of challenging decision making and control tasks. However, these
        methods typically suffer from two major challenges: very high sample complexity
        and brittle convergence properties, which necessitate meticulous hyperparameter
        tuning. Both of these challenges severely limit the applicability of such
        methods to complex, real-world domains. In this paper, we propose soft\nactor-critic, an off-policy actor-critic deep RL algorithm based on the maximum\nentropy reinforcement learning framework. In this framework, the actor aims to\nmaximize expected reward while also maximizing entropy. That is, to succeed at\nthe task while acting as randomly as possible. Prior deep RL methods based on\nthis framework have been formulated as Q-learning methods. By combining\noff-policy updates with a stable stochastic actor-critic formulation, our\nmethod achieves state-of-the-art performance on a range of continuous control\nbenchmark tasks, outperforming prior on-policy and off-policy methods.\nFurthermore, we demonstrate that, in contrast to other off-policy algorithms,\nour approach is very stable, achieving very similar performance across\ndifferent random seeds.""",
        extra='',
        priority=0.0,
        tags=set([]),
        authors=[u'Tuomas Haarnoja', u'Aurick Zhou', u'Pieter Abbeel', u'Sergey Levine'],
        # arxiv_id='1801.01290',
        read=False, url='')
]
