import attr


@attr.s
class Paper(object):
    title = attr.ib(type=str, default="")
    desc = attr.ib(type=str, default="")
    extra = attr.ib(type=str, default="")
    rating = attr.ib(type=int, default=0)
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
    )
]
