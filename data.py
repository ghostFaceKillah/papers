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
        rating=4,
        priority=0.0
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
        title='Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor',
        desc="""Model-free deep reinforcement learning (RL) algorithms have been demonstrated
        on a range of challenging decision making and control tasks. However, these
        methods typically suffer from two major challenges: very high sample complexity
        and brittle convergence properties, which necessitate meticulous hyperparameter
        tuning. Both of these challenges severely limit the applicability of such
        methods to complex, real-world domains. In this paper, we propose soft
        actor-critic, an off-policy actor-critic deep RL algorithm based on the maximum
        entropy reinforcement learning framework. In this framework, the actor aims to
        maximize expected reward while also maximizing entropy. That is, to succeed at
        the task while acting as randomly as possible. Prior deep RL methods based on
        this framework have been formulated as Q-learning methods. By combining
        off-policy updates with a stable stochastic actor-critic formulation, our
        method achieves state-of-the-art performance on a range of continuous control
        benchmark tasks, outperforming prior on-policy and off-policy methods.
        Furthermore, we demonstrate that, in contrast to other off-policy algorithms,
        our approach is very stable, achieving very similar performance across
        different random seeds.""",
        extra='',
        priority=8.0,
        rating=9,
        tags=set([]),
        authors=[u'Tuomas Haarnoja', u'Aurick Zhou', u'Pieter Abbeel', u'Sergey Levine'],
        # arxiv_id='1801.01290',
        read=False,
        url=''
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
        authors=['Danijar Hafner', 'Timothy Lillicrap',
                 'Ian Fischer', 'Ruben Villegas', 'David Ha', 'Honglak Lee', 'James Davidson'],
        arxiv_id='1811.04551',
        read=True,
        url='https://planetrl.github.io'
    ),
    Paper(title='Insights into LSTM Fully Convolutional Networks for Time Series  Classification',
          desc='Long Short Term Memory Fully Convolutional Neural Networks (LSTM-FCN) and'
               'Attention LSTM-FCN (ALSTM-FCN) have shown to achieve state-of-the-art\n'
               'performance on the task of classifying time series signals on the old\n'
               'University of California-Riverside (UCR) time series repository. However, there\n'
               'has been no study on why LSTM-FCN and ALSTM-FCN perform well. In this paper, we\n'
               'perform a series of ablation tests (3627 experiments) on LSTM-FCN and ALSTM-FCN\n'
               'to provide a better understanding of the model and each of its sub-module.\n'
               'Results from the ablation tests on ALSTM-FCN and LSTM-FCN show that the these'
               'blocks perform better when applied in a conjoined manner. Two z-normalizing'
               'techniques, z-normalizing each sample independently and z-normalizing the whole'
               'dataset, are compared using a Wilcoxson signed-rank test to show a statistical'
               'difference in performance. In addition, we provide an understanding of the'
               'impact dimension shuffle has on LSTM-FCN by comparing its performance with'
               'LSTM-FCN when no dimension shuffle is applied. Finally, we demonstrate the'
               'performance of the LSTM-FCN when the LSTM block is replaced by a GRU, basic'
               'RNN, and Dense Block.',
          extra='',
          rating=0,
          priority=0.0,
          tags=set(),
          authors=['Fazle Karim', 'Somshubra Majumdar', 'Houshang Darabi'],
          arxiv_id='1902.10756',
          read=False,
          url=''
    ),
    Paper(title='1D Convolutional Neural Network Models for Sleep Arousal Detection',
          desc='Sleep arousals transition the depth of sleep to a more superficial stage. The'
               'occurrence of such events is often considered as a protective mechanism to'
               'alert the body of harmful stimuli. Thus, accurate sleep arousal detection can'
               'lead to an enhanced understanding of the underlying causes and influencing the'
               'assessment of sleep quality. Previous studies and guidelines have suggested'
               'that sleep arousals are linked mainly to abrupt frequency shifts in EEG'
               'signals, but the proposed rules are shown to be insufficient for a'
               'comprehensive characterization of arousals. This study investigates the'
               'application of five recent convolutional neural networks (CNNs) for sleep'
               'arousal detection and performs comparative evaluations to determine the best'
               'model for this task. The investigated state-of-the-art CNN models have'
               'originally been designed for image or speech processing. A detailed set of'
               'evaluations is performed on the benchmark dataset provided by'
               'PhysioNet/Computing in Cardiology Challenge 2018, and the results show that the'
               'best 1D CNN model has achieved an average of 0.31 and 0.84 for the area under'
               'the precision-recall and area under the ROC curves, respectively.',
          extra='',
          rating=0,
          priority=0.0,
          tags=set(),
          authors=['Morteza Zabihi', 'Ali Bahrami Rad', 'Serkan Kiranyaz',
                   'Simo Särkkä', 'Moncef Gabbouj'],
          arxiv_id='1903.01552', read=False, url=''),
    Paper(
        title='FickleNet: Weakly and Semi-supervised Semantic Image Segmentation\\\\using'
              '  Stochastic Inference',
        desc='The main obstacle to weakly supervised semantic image segmentation is the'
             'difficulty of obtaining pixel-level information from coarse image-level'
             'annotations. Most methods based on image-level annotations use localization'
             'maps obtained from the classifier, but these only focus on the small'
             'discriminative parts of objects and do not capture precise boundaries.'
             'FickleNet explores diverse combinations of locations on feature maps created by'
             'generic deep neural networks. It selects hidden units randomly and then uses'
             'them to obtain activation scores for image classification. FickleNet implicitly'
             'learns the coherence of each location in the feature maps, resulting in a'
             'localization map which identifies both discriminative and other parts of'
             'objects. The ensemble effects are obtained from a single network by selecting'
             'random hidden unit pairs, which means that a variety of localization maps are'
             'generated from a single image. Our approach does not require any additional'
             'training steps and only adds a simple layer to a standard convolutional neural'
             'network; nevertheless it outperforms recent comparable techniques on the Pascal'
             'VOC 2012 benchmark in both weakly and semi-supervised settings.',
        extra='',
        rating=0,
        priority=0.0,
        tags=set(),
        authors=['Jungbeom Lee', 'Eunji Kim', 'Sungmin Lee', 'Jangho Lee', 'Sungroh Yoon'],
        arxiv_id='1902.10421v1',
        read=False,
        url=''
    ),
    Paper(title='Stochastically Rank-Regularized Tensor Regression Networks', desc='Over-parametrization of deep neural networks has recently been shown to be'
                                                                                   'key to their successful training. However, it also renders them prone to\noverfitting and makes them expensive to store and train. Tensor regression\nnetworks significantly reduce the number of effective parameters in deep neural\nnetworks while retaining accuracy and the ease of training. They replace the\nflattening and fully-connected layers with a tensor regression layer, where the\nregression weights are expressed through the factors of a low-rank tensor\ndecomposition. In this paper, to further improve tensor regression networks, we\npropose a novel stochastic rank-regularization. It consists of a novel\nrandomized tensor sketching method to approximate the weights of tensor\nregression layers. We theoretically and empirically establish the link between\nour proposed stochastic rank-regularization and the dropout on low-rank tensor\nregression. Extensive experimental results with both synthetic data and real\nworld datasets (i.e., CIFAR-100 and the UK Biobank brain MRI dataset) support\nthat the proposed approach i) improves performance in both classification and\nregression tasks, ii) decreases overfitting, iii) leads to more stable training\nand iv) improves robustness to adversarial attacks and random noise.', extra='', rating=0, priority=0.0, tags=set(), authors=['Arinbjörn Kolbeinsson', 'Jean Kossaifi', 'Yannis Panagakis', 'Anima Anandkumar', 'Ioanna Tzoulaki', 'Paul Matthews'], arxiv_id='1902.10758v1', read=False, url=''),
    Paper(title='TraVeLGAN: Image-to-image Translation by Transformation Vector Learning', desc='Interest in image-to-image translation has grown substantially in recent\nyears with the success of unsupervised models based on the cycle-consistency\nassumption. The achievements of these models have been limited to a particular\nsubset of domains where this assumption yields good results, namely homogeneous\ndomains that are characterized by style or texture differences. We tackle the\nchallenging problem of image-to-image translation where the domains are defined\nby high-level shapes and contexts, as well as including significant clutter and\nheterogeneity. For this purpose, we introduce a novel GAN based on preserving\nintra-domain vector transformations in a latent space learned by a siamese\nnetwork. The traditional GAN system introduced a discriminator network to guide\nthe generator into generating images in the target domain. To this two-network\nsystem we add a third: a siamese network that guides the generator so that each\noriginal image shares semantics with its generated version. With this new\nthree-network system, we no longer need to constrain the generators with the\nubiquitous cycle-consistency restraint. As a result, the generators can learn\nmappings between more complex domains that differ from each other by large\ndifferences - not just style or texture.', extra='', rating=0, priority=0.0, tags=set(), authors=['Matthew Amodio', 'Smita Krishnaswamy'], arxiv_id='1902.09631v1', read=False, url=''),
    Paper(title='FickleNet: Weakly and Semi-supervised Semantic Image Segmentation\\\\using\n  Stochastic Inference', desc='The main obstacle to weakly supervised semantic image segmentation is the\ndifficulty of obtaining pixel-level information from coarse image-level\nannotations. Most methods based on image-level annotations use localization\nmaps obtained from the classifier, but these only focus on the small\ndiscriminative parts of objects and do not capture precise boundaries.\nFickleNet explores diverse combinations of locations on feature maps created by\ngeneric deep neural networks. It selects hidden units randomly and then uses\nthem to obtain activation scores for image classification. FickleNet implicitly\nlearns the coherence of each location in the feature maps, resulting in a\nlocalization map which identifies both discriminative and other parts of\nobjects. The ensemble effects are obtained from a single network by selecting\nrandom hidden unit pairs, which means that a variety of localization maps are\ngenerated from a single image. Our approach does not require any additional\ntraining steps and only adds a simple layer to a standard convolutional neural\nnetwork; nevertheless it outperforms recent comparable techniques on the Pascal\nVOC 2012 benchmark in both weakly and semi-supervised settings.', extra='', rating=0, priority=0.0, tags=set(), authors=['Jungbeom Lee', 'Eunji Kim', 'Sungmin Lee', 'Jangho Lee', 'Sungroh Yoon'], arxiv_id='1902.10421v1', read=False, url=''),
    Paper(title='Model-Based Active Exploration', url='https://drive.google.com/file/d/0B_utB5Y8Y6D5YmJVaklhVlc3RVQ5V0VSelg2dC1xZUY4MjFF/view'),
    Paper(title='Photorealistic Style Transfer via Wavelet Transforms', desc='Recent style transfer models have provided promising artistic results.\nHowever, given a photograph as a reference style, existing methods are limited\nby spatial distortions or unrealistic artifacts, which should not happen in\nreal photographs. We introduce a theoretically sound correction to the network\narchitecture that remarkably enhances photorealism and faithfully transfers the\nstyle. The key ingredient of our method is wavelet transforms that naturally\nfits in deep networks. We propose a wavelet corrected transfer based on\nwhitening and coloring transforms (WCT$^2$) that allows features to preserve\ntheir structural information and statistical properties of VGG feature space\nduring stylization. This is the first and the only end-to-end model that can\nstylize $1024\\times1024$ resolution image in 4.7 seconds, giving a pleasing and\nphotorealistic quality without any post-processing. Last but not least, our\nmodel provides a stable video stylization without temporal constraints. The\ncode, generated images, supplementary materials, and pre-trained models are all\navailable at https://github.com/ClovaAI/WCT2.', extra='', rating=0, priority=0.0, tags=set(), authors=['Jaejun Yoo', 'Youngjung Uh', 'Sanghyuk Chun', 'Byeongkyu Kang', 'Jung-Woo Ha'], arxiv_id='1903.09760v1', read=False, url=''),
    Paper(
        title='DeepRED: Deep Image Prior Powered by RED',
        desc='Inverse problems in imaging are extensively studied, with a variety of'
             'strategies, tools, and theory that have been accumulated over the years.'
             'Recently, this field has been immensely influenced by the emergence of'
             'deep-learning techniques. One such contribution, which is the focus of this'
             'paper, is the Deep Image Prior (DIP) work by Ulyanov, Vedaldi, and Lempitsky'
             '(2018). DIP offers a new approach towards the regularization of inverse'
             'problems, obtained by forcing the recovered image to be synthesized from a'
             'given deep architecture. While DIP has been shown to be effective, its results'
             'fall short when compared to state-of-the-art alternatives. In this work, we aim'
             'to boost DIP by adding an explicit prior, which enriches the overall'
             'regularization effect in order to lead to better-recovered images. More'
             'specifically, we propose to bring-in the concept of Regularization by Denoising'
             '(RED), which leverages existing denoisers for regularizing inverse problems.'
             'Our work shows how the two (DeepRED) can be merged to a highly effective'
             'recovery process while avoiding the need to differentiate the chosen denoiser,'
             'and leading to very effective results, demonstrated for several tested inverse'
             'problems.',
        extra='',
        rating=0,
        priority=0.0,
        tags=set(),
        authors=['Gary Mataev', 'Michael Elad', 'Peyman Milanfar'],
        arxiv_id='1903.10176v1',
        read=False,
        url=''
    ),
    Paper(title='Spatiotemporal Feature Learning for Event-Based Vision', desc='Unlike conventional frame-based sensors, event-based visual sensors output\ninformation through spikes at a high temporal resolution. By only encoding\nchanges in pixel intensity, they showcase a low-power consuming, low-latency\napproach to visual information sensing. To use this information for higher\nsensory tasks like object recognition and tracking, an essential simplification\nstep is the extraction and learning of features. An ideal feature descriptor\nmust be robust to changes involving (i) local transformations and (ii)\nre-appearances of a local event pattern. To that end, we propose a novel\nspatiotemporal feature representation learning algorithm based on slow feature\nanalysis (SFA). Using SFA, smoothly changing linear projections are learnt\nwhich are robust to local visual transformations. In order to determine if the\nfeatures can learn to be invariant to various visual transformations, feature\npoint tracking tasks are used for evaluation. Extensive experiments across two\ndatasets demonstrate the adaptability of the spatiotemporal feature learner to\ntranslation, scaling and rotational transformations of the feature points. More\nimportantly, we find that the obtained feature representations are able to\nexploit the high temporal resolution of such event-based cameras in generating\nbetter feature tracks.', extra='', rating=0, priority=0.0, tags=set(), authors=['Rohan Ghosh', 'Anupam Gupta', 'Siyi Tang', 'Alcimar Soares', 'Nitish Thakor'], arxiv_id='1903.06923v1', read=False, url=''),
    Paper(title='Learning-Based Animation of Clothing for Virtual Try-On', desc='This paper presents a learning-based clothing animation method for highly\nefficient virtual try-on simulation. Given a garment, we preprocess a rich\ndatabase of physically-based dressed character simulations, for multiple body\nshapes and animations. Then, using this database, we train a learning-based\nmodel of cloth drape and wrinkles, as a function of body shape and dynamics. We\npropose a model that separates global garment fit, due to body shape, from\nlocal garment wrinkles, due to both pose dynamics and body shape. We use a\nrecurrent neural network to regress garment wrinkles, and we achieve highly\nplausible nonlinear effects, in contrast to the blending artifacts suffered by\nprevious methods. At runtime, dynamic virtual try-on animations are produced in\njust a few milliseconds for garments with thousands of triangles. We show\nqualitative and quantitative analysis of results', extra='', rating=0, priority=0.0, tags=set(), authors=['Igor Santesteban', 'Miguel A. Otaduy', 'Dan Casas'], arxiv_id='1903.07190v1', read=False, url=''),
    Paper(title='Smart, Deep Copy-Paste', desc='In this work, we propose a novel system for smart copy-paste, enabling the\nsynthesis of high-quality results given a masked source image content and a\ntarget image context as input. Our system naturally resolves both shading and\ngeometric inconsistencies between source and target image, resulting in a\nmerged result image that features the content from the pasted source image,\nseamlessly pasted into the target context. Our framework is based on a novel\ntraining image transformation procedure that allows to train a deep\nconvolutional neural network end-to-end to automatically learn a representation\nthat is suitable for copy-pasting. Our training procedure works with any image\ndataset without additional information such as labels, and we demonstrate the\neffectiveness of our system on two popular datasets, high-resolution face\nimages and the more complex Cityscapes dataset. Our technique outperforms the\ncurrent state of the art on face images, and we show promising results on the\nCityscapes dataset, demonstrating that our system generalizes to much higher\nresolution than the training data.', extra='', rating=0, priority=0.0, tags=set(), authors=['Tiziano Portenier', 'Qiyang Hu', 'Paolo Favaro', 'Matthias Zwicker'], arxiv_id='1903.06763v1', read=False, url=''),
    Paper(
        title='Relational inductive biases, deep learning, and graph networks',
        desc='Artificial intelligence (AI) has undergone a renaissance recently, making'
             'major progress in key domains such as vision, language, control, and'
             'decision-making. This has been due, in part, to cheap data and cheap compute'
             'resources, which have fit the natural strengths of deep learning. However, many'
             'defining characteristics of human intelligence, which developed under much'
             'different pressures, remain out of reach for current approaches. In particular,'
             'generalizing beyond one\'s experiences--a hallmark of human intelligence from'
             'infancy--remains a formidable challenge for modern AI.'
             '  The following is part position paper, part review, and part unification. We'
             'argue that combinatorial generalization must be a top priority for AI to'
             'achieve human-like abilities, and that structured representations and'
             'computations are key to realizing this objective. Just as biology uses nature'
             'and nurture cooperatively, we reject the false choice between'
             '"hand-engineering" and "end-to-end" learning, and instead advocate for an'
             'approach which benefits from their complementary strengths. We explore how'
             'using relational inductive biases within deep learning architectures can'
             'facilitate learning about entities, relations, and rules for composing them. We'
             'present a new building block for the AI toolkit with a strong relational'
             'inductive bias--the graph network--which generalizes and extends various'
             'approaches for neural networks that operate on graphs, and provides a'
             'straightforward interface for manipulating structured knowledge and producing'
             'structured behaviors. We discuss how graph networks can support relational'
             'reasoning and combinatorial generalization, laying the foundation for more'
             'sophisticated, interpretable, and flexible patterns of reasoning. As a'
             'companion to this paper, we have released an open-source software library for'
             'building graph networks, with demonstrations of how to use them in practice.',
        extra='',
        rating=0,
        priority=0.0,
        tags=set(),
        authors=['Peter W. Battaglia', 'Jessica B. Hamrick', 'Victor Bapst', 'Alvaro Sanchez-Gonzalez',
                 'Vinicius Zambaldi', 'Mateusz Malinowski', 'Andrea Tacchetti', 'David Raposo',
                 'Adam Santoro', 'Ryan Faulkner', 'Caglar Gulcehre', 'Francis Song', 'Andrew Ballard',
                 'Justin Gilmer', 'George Dahl', 'Ashish Vaswani', 'Kelsey Allen', 'Charles Nash',
                 'Victoria Langston', 'Chris Dyer', 'Nicolas Heess', 'Daan Wierstra', 'Pushmeet Kohli',
                 'Matt Botvinick', 'Oriol Vinyals', 'Yujia Li', 'Razvan Pascanu'],
        arxiv_id='1806.01261', read=False, url=''
    ),
    Paper(
        title='Guided Policy Search',
        url="https://graphics.stanford.edu/projects/gpspaper/gps_full.pdf",
        desc='Direct policy search can effectively scale to high-dimensional systems, but complex'
             'policies with hundreds of parameters often' 'present a challenge for such methods,'
             ' requiring numerous samples and often falling into' 'poor local optima.'
             ' We present a guided policy search algorithm that uses trajectory optimization'
             ' to direct policy learning and avoid' 'poor local optima. We show how differential'
             'dynamic programming can be used to generate suitable guiding samples, and describe a'
             'regularized importance sampled policy optimization that incorporates these samples into'
             'the policy search. We evaluate the method by'
             'learning neural network controllers for planar'
             'swimming, hopping, and walking, as well as'
             'simulated 3D humanoid running.',
        priority=9
    ),
    Paper(
        title='Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning',
        priority=2
    )

]
