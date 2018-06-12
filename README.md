# Supervised Data Synthesizing and Evolving
A method to handle the *High Imbalanced and Low Separable* data. 

* Balance the majorities and minorities:
    * *NearMiss* under-sampling and *Borderline-SMOTE* over-sampling are employed in a hybrid manner. 
    * *Differential Evolution* is used to 'evolve' the data under certain supervision.
* Learn from the balanced data:
    * An MLP is customized with *batch-normalization* and *dropout*.
* Result:
    * Significantly higher *F-Measure* and *G-Mean* value than state-of-the-art algorithms.