# classify-breat-cancer.py
# 17-jan-2020
# This code demonstrates decision tree classification on
# the Wisconsin breast cancer data set

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier


# parameters
plot_step = 0.02
plot_colors = "rm"
classes = 2 # has cancer (1) or doesn't have it (0)


# load data
breast_cancer = load_breast_cancer()
num_features = len( breast_cancer.data[ 0 ] )

# classify and plot data
plt.figure()
plt.rc( 'xtick', labelsize = 8 )
plt.rc( 'ytick', labelsize = 8 )

# pair up features and plot their graphs
for i in range( 0, num_features ):
    for j in range( i + 1, num_features ):
        pair = [ i, j ]
        features = breast_cancer.data[:, pair] # splits the list and picks out 
                                          # the features specified in the pair
        answers = breast_cancer.target
        # train classifier
        tree = DecisionTreeClassifier().fit( features, answers )
        # plot decision boundaries by enumerating from
        # the min value to the max value observed from the data
        plt.subplot( num_features, num_features, j*num_features + i + 1 )
        x_min, x_max = features[ :, 0 ].min() - 1, features[ :, 0 ].max() + 1
        y_min, y_max = features[ :, 1 ].min() - 1, features[ :, 1 ].max() + 1
        xx, yy = np.meshgrid( np.arange( x_min, x_max, plot_step ), 
                              np.arange( y_min, y_max, plot_step ) )
        Z = tree.predict( np.c_[ xx.ravel(), yy.ravel() ] )
        Z = Z.reshape( xx.shape )
        cs = plt.contourf( xx, yy, Z, cmap = plt.cm.Paired )
        plt.xlabel( breast_cancer.feature_names[ pair[ 0 ] ], fontsize = 8 )
        plt.ylabel( breast_cancer.feature_names[ pair[ 1 ] ], fontsize = 8 )
        plt.axis( "tight" )
        # plot training points
        for ii, colour in zip( range( classes ), plot_colors ):
            idx = np.where( answers == ii )
            plt.scatter( features[ idx, 0 ], features[ idx, 1 ], c = colour, 
                         label = breast_cancer.target_names[ ii ], cmap = plt.cm.Paired )
            plt.axis( "tight" )

plt.show()