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
        read=False, url=''
    ),
    Paper(
        title='Learning Latent Dynamics for Planning from Pixels',
        desc="""Original abstract: Planning has been very successful for control tasks with known environment
               dynamics. To leverage planning in unknown environments, the agent needs to
               learn the dynamics from interactions with the world. However, learning dynamics
               models that are accurate enough for planning has been a long-standing
               challenge, especially in image-based domains. We propose the Deep Planning
               Network (PlaNet), a purely model-based agent that learns the environment
               dynamics from images and chooses actions through fast online planning in latent
               space. To achieve high performance, the dynamics model must accurately predict
               the rewards ahead for multiple time steps. We approach this problem using a
               latent dynamics model with both deterministic and stochastic transition
               components and a multi-step variational inference objective that we call latent
               overshooting. Using only pixel observations, our agent solves continuous
               control tasks with contact dynamics, partial observability, and sparse rewards,
               which exceed the difficulty of tasks that were previously solved by planning
               with learned models. PlaNet uses substantially fewer episodes and reaches final
               performance close to and sometimes higher than strong model-free algorithms.
               
               My notes:
               Learn a model - dynamics in the latent space + reward from latent using RNN / VAE style architecture.
               No policy - instead use CEM in the latent space.
               Similar compute, but 50 times more env efficient than policy gradient (a3c, D4PG) 
               for good performance on mujoco.
               Open source code.
               
               """,
        extra="""""",
        rating=9,
        priority=0.6,
        tags=set(),
        authors=['Danijar Hafner', 'Timothy Lillicrap', 'Ian Fischer', 'Ruben Villegas', 'David Ha', 'Honglak Lee', 'James Davidson'],
        arxiv_id='1811.04551',
        read=True,
        url='https://planetrl.github.io'
    ),
    Paper(title='Insights into LSTM Fully Convolutional Networks for Time Series\n  Classification', desc='Long Short Term Memory Fully Convolutional Neural Networks (LSTM-FCN) and\nAttention LSTM-FCN (ALSTM-FCN) have shown to achieve state-of-the-art\nperformance on the task of classifying time series signals on the old\nUniversity of California-Riverside (UCR) time series repository. However, there\nhas been no study on why LSTM-FCN and ALSTM-FCN perform well. In this paper, we\nperform a series of ablation tests (3627 experiments) on LSTM-FCN and ALSTM-FCN\nto provide a better understanding of the model and each of its sub-module.\nResults from the ablation tests on ALSTM-FCN and LSTM-FCN show that the these\nblocks perform better when applied in a conjoined manner. Two z-normalizing\ntechniques, z-normalizing each sample independently and z-normalizing the whole\ndataset, are compared using a Wilcoxson signed-rank test to show a statistical\ndifference in performance. In addition, we provide an understanding of the\nimpact dimension shuffle has on LSTM-FCN by comparing its performance with\nLSTM-FCN when no dimension shuffle is applied. Finally, we demonstrate the\nperformance of the LSTM-FCN when the LSTM block is replaced by a GRU, basic\nRNN, and Dense Block.', extra='', rating=0, priority=0.0, tags=set(), authors=['Fazle Karim', 'Somshubra Majumdar', 'Houshang Darabi'], arxiv_id='1902.10756', read=False, url=''),
    Paper(title='1D Convolutional Neural Network Models for Sleep Arousal Detection', desc='Sleep arousals transition the depth of sleep to a more superficial stage. The\noccurrence of such events is often considered as a protective mechanism to\nalert the body of harmful stimuli. Thus, accurate sleep arousal detection can\nlead to an enhanced understanding of the underlying causes and influencing the\nassessment of sleep quality. Previous studies and guidelines have suggested\nthat sleep arousals are linked mainly to abrupt frequency shifts in EEG\nsignals, but the proposed rules are shown to be insufficient for a\ncomprehensive characterization of arousals. This study investigates the\napplication of five recent convolutional neural networks (CNNs) for sleep\narousal detection and performs comparative evaluations to determine the best\nmodel for this task. The investigated state-of-the-art CNN models have\noriginally been designed for image or speech processing. A detailed set of\nevaluations is performed on the benchmark dataset provided by\nPhysioNet/Computing in Cardiology Challenge 2018, and the results show that the\nbest 1D CNN model has achieved an average of 0.31 and 0.84 for the area under\nthe precision-recall and area under the ROC curves, respectively.', extra='', rating=0, priority=0.0, tags=set(), authors=['Morteza Zabihi', 'Ali Bahrami Rad', 'Serkan Kiranyaz', 'Simo Särkkä', 'Moncef Gabbouj'], arxiv_id='1903.01552', read=False, url=''),
    Paper(title='FickleNet: Weakly and Semi-supervised Semantic Image Segmentation\\\\using\n  Stochastic Inference', desc='The main obstacle to weakly supervised semantic image segmentation is the\ndifficulty of obtaining pixel-level information from coarse image-level\nannotations. Most methods based on image-level annotations use localization\nmaps obtained from the classifier, but these only focus on the small\ndiscriminative parts of objects and do not capture precise boundaries.\nFickleNet explores diverse combinations of locations on feature maps created by\ngeneric deep neural networks. It selects hidden units randomly and then uses\nthem to obtain activation scores for image classification. FickleNet implicitly\nlearns the coherence of each location in the feature maps, resulting in a\nlocalization map which identifies both discriminative and other parts of\nobjects. The ensemble effects are obtained from a single network by selecting\nrandom hidden unit pairs, which means that a variety of localization maps are\ngenerated from a single image. Our approach does not require any additional\ntraining steps and only adds a simple layer to a standard convolutional neural\nnetwork; nevertheless it outperforms recent comparable techniques on the Pascal\nVOC 2012 benchmark in both weakly and semi-supervised settings.', extra='', rating=0, priority=0.0, tags=set(), authors=['Jungbeom Lee', 'Eunji Kim', 'Sungmin Lee', 'Jangho Lee', 'Sungroh Yoon'], arxiv_id='1902.10421v1', read=False, url=''),
    Paper(title='Stochastically Rank-Regularized Tensor Regression Networks', desc='Over-parametrization of deep neural networks has recently been shown to be\nkey to their successful training. However, it also renders them prone to\noverfitting and makes them expensive to store and train. Tensor regression\nnetworks significantly reduce the number of effective parameters in deep neural\nnetworks while retaining accuracy and the ease of training. They replace the\nflattening and fully-connected layers with a tensor regression layer, where the\nregression weights are expressed through the factors of a low-rank tensor\ndecomposition. In this paper, to further improve tensor regression networks, we\npropose a novel stochastic rank-regularization. It consists of a novel\nrandomized tensor sketching method to approximate the weights of tensor\nregression layers. We theoretically and empirically establish the link between\nour proposed stochastic rank-regularization and the dropout on low-rank tensor\nregression. Extensive experimental results with both synthetic data and real\nworld datasets (i.e., CIFAR-100 and the UK Biobank brain MRI dataset) support\nthat the proposed approach i) improves performance in both classification and\nregression tasks, ii) decreases overfitting, iii) leads to more stable training\nand iv) improves robustness to adversarial attacks and random noise.', extra='', rating=0, priority=0.0, tags=set(), authors=['Arinbjörn Kolbeinsson', 'Jean Kossaifi', 'Yannis Panagakis', 'Anima Anandkumar', 'Ioanna Tzoulaki', 'Paul Matthews'], arxiv_id='1902.10758v1', read=False, url=''),
    Paper(title='TraVeLGAN: Image-to-image Translation by Transformation Vector Learning', desc='Interest in image-to-image translation has grown substantially in recent\nyears with the success of unsupervised models based on the cycle-consistency\nassumption. The achievements of these models have been limited to a particular\nsubset of domains where this assumption yields good results, namely homogeneous\ndomains that are characterized by style or texture differences. We tackle the\nchallenging problem of image-to-image translation where the domains are defined\nby high-level shapes and contexts, as well as including significant clutter and\nheterogeneity. For this purpose, we introduce a novel GAN based on preserving\nintra-domain vector transformations in a latent space learned by a siamese\nnetwork. The traditional GAN system introduced a discriminator network to guide\nthe generator into generating images in the target domain. To this two-network\nsystem we add a third: a siamese network that guides the generator so that each\noriginal image shares semantics with its generated version. With this new\nthree-network system, we no longer need to constrain the generators with the\nubiquitous cycle-consistency restraint. As a result, the generators can learn\nmappings between more complex domains that differ from each other by large\ndifferences - not just style or texture.', extra='', rating=0, priority=0.0, tags=set(), authors=['Matthew Amodio', 'Smita Krishnaswamy'], arxiv_id='1902.09631v1', read=False, url=''),
    Paper(title='FickleNet: Weakly and Semi-supervised Semantic Image Segmentation\\\\using\n  Stochastic Inference', desc='The main obstacle to weakly supervised semantic image segmentation is the\ndifficulty of obtaining pixel-level information from coarse image-level\nannotations. Most methods based on image-level annotations use localization\nmaps obtained from the classifier, but these only focus on the small\ndiscriminative parts of objects and do not capture precise boundaries.\nFickleNet explores diverse combinations of locations on feature maps created by\ngeneric deep neural networks. It selects hidden units randomly and then uses\nthem to obtain activation scores for image classification. FickleNet implicitly\nlearns the coherence of each location in the feature maps, resulting in a\nlocalization map which identifies both discriminative and other parts of\nobjects. The ensemble effects are obtained from a single network by selecting\nrandom hidden unit pairs, which means that a variety of localization maps are\ngenerated from a single image. Our approach does not require any additional\ntraining steps and only adds a simple layer to a standard convolutional neural\nnetwork; nevertheless it outperforms recent comparable techniques on the Pascal\nVOC 2012 benchmark in both weakly and semi-supervised settings.', extra='', rating=0, priority=0.0, tags=set(), authors=['Jungbeom Lee', 'Eunji Kim', 'Sungmin Lee', 'Jangho Lee', 'Sungroh Yoon'], arxiv_id='1902.10421v1', read=False, url=''),
    Paper(title='Model-Based Active Exploration', url='https://drive.google.com/file/d/0B_utB5Y8Y6D5YmJVaklhVlc3RVQ5V0VSelg2dC1xZUY4MjFF/view')
]
