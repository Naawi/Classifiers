# classify-breast-cancer-simple.py
# 17-jan-2020
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# Parameters
plot_step = 0.02


def build_scatter_plot( ftr_1, ftr_2 ):
    # Load the data
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data[:, [ ftr_1, ftr_2 ] ] 
    y = breast_cancer.target 

    breast_cancer_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X, y) 

    # test accuracy of model
    # X_train, X_test, y_train, y_test = train_test_split( breast_cancer.data, breast_cancer.target, 
    #                                                      test_size = 0.1, random_state = 0 )
    # breast_cancer_tree = tree.DecisionTreeClassifier( criterion = "entropy" ).fit( X_train, y_train ) 
    # print( "accuracy:", breast_cancer_tree.score( X_test, y_test ) )

    # Now plot the decision surface that we just learnt by using the decision tree to
    # classify every background point.
    # 0 is the first feature, 1 is the other feature
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                        np.arange(y_min, y_max, plot_step))

    Z = breast_cancer_tree.predict(np.c_[xx.ravel(), yy.ravel()]) # Here we use the tree
                                                        # to predict the classification
                                                        # of each background point.
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    # Also plot the original data on the same axes
    plt.scatter(X[:, 0], X[:, 1], c=y.astype(np.float))

    # Label axes
    plt.xlabel( breast_cancer.feature_names[ ftr_1 ], fontsize=10 )
    plt.ylabel( breast_cancer.feature_names[ ftr_2 ], fontsize=10 )

    #plt.savefig( f"breast-cancer-plts/features-{ftr_1}-{ftr_2}.png" )
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Specify features for building decision tree')
    parser.add_argument('--f1', metavar = 'int', required = True )
    parser.add_argument('--f2', metavar = 'int', required = True )
    args = parser.parse_args()
    build_scatter_plot( ftr_1 = int( args.f1 ), ftr_2 = int( args.f2 ) )