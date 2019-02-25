import attr


@attr.s
class Paper(object):
    desc = attr.ib(type=str, default="")
    extra = attr.ib(type=str, default="")
    tags = attr.ib(type=set, factory=set)
    authors = attr.ib(type=list, factory=list)
    arxiv_id = attr.ib(type=str, default="")


DB = [
    Paper( 
        desc=u"""
            We propose an expert-augmented actor-critic algorithm, which
            we evaluate on\ntwo environments with sparse rewards: Montezumas Revenge
            and a demanding maze\nfrom the ViZDoom suite. In the case of Montezumas
            Revenge, an agent trained\nwith our method achieves very good results
            consistently scoring above 27,000\npoints (in many experiments beating the
                first world). With an appropriate\nchoice of hyperparameters, our
            algorithm surpasses the performance of the\nexpert data. In a number of
            experiments, we have observed an unreported bug in\nMontezumas Revenge
            which allowed the agent to score more than 800,000 points.
        """,
        extra='',
        tags=set('rl'), 
        authors=[
            u'Michal Garmulewicz', u'Henryk Michalewski', u'Piotr Milos'
        ],
        arxiv_id='1809.03447'
    ),

]
