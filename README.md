# Naive Bayes text classifiers: a locally weighted learning approach, Liangxiao Jianga, Zhihua Caia*, Harry Zhangb and Dianhong Wangc.
Abstract:
Due to being fast, easy to implement and relatively effective, some state-of-the-art
naive Bayes text classifiers with the strong assumption of conditional independence
among attributes, such as multinomial naive Bayes, complement naive
Bayes and the one-versus-all-but-one model, have received a great deal of
attention from researchers in the domain of text classification. In this article, we
revisit these naive Bayes text classifiers and empirically compare their classification
performance on a large number of widely used text classification benchmark
datasets. Then, we propose a locally weighted learning approach to these naive
Bayes text classifiers. We call our new approach locally weighted naive Bayes text
classifiers (LWNBTC). LWNBTC weakens the attribute conditional independence
assumption made by these naive Bayes text classifiers by applying the
locally weighted learning approach. The experimental results show that our
locally weighted versions significantly outperform these state-of-the-art naive
Bayes text classifiers in terms of classification accuracy.
......................................................................................................

I need help to implement LWMNB, and LWCNB only.
i implement LWMNB but the results i got worse than in paper.
Platform: weka with java


    1. paper
    2. Distance metric (textDis.java place it under core ), and updatableDistance.java under core
    3. LinearSearch2.java (place under core/ neighboursearch)
    4. LWMnb.java (place under classifier/bayes)
    5. textClassification_ecl.java ( test java class place under default package).
    
