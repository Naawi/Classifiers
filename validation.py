from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn import metrics
from statistics import stdev



def getAccuracy( data_set, split ):
    X_train, X_test, y_train, y_test = train_test_split( data_set.data, data_set.target, 
                                                         test_size = split, random_state = 1 ) # what should I set the seed to?
    dtree = tree.DecisionTreeClassifier( criterion = "entropy" ).fit( X_train, y_train ) 
    score = dtree.score( X_test, y_test )
    return score


def manual_cross_validation( data_set, split, runs = 10 ):
    scores = []
    for _ in range( runs ):
        scores.append( getAccuracy( data_set, split ) )

    avg = sum( scores ) / runs
    sd = stdev( scores )

    print("mean: ", avg )
    print("stdev: ", sd )


def cross_validation( data_set ):
    dtree = tree.DecisionTreeClassifier( criterion = "entropy" ).fit( data_set.data, data_set.target )
    cv_scores = cross_val_score( dtree, data_set.data, 
                                 data_set.target, cv = 10 )
    print( "mean:", cv_scores.mean() )
    print( "stdev:", cv_scores.std() )



if __name__ == "__main__":
    print( "manual cross validation:" )
    averageScore( load_breast_cancer(), 0.1 )

    print( "using provided cross validation module:" )
    cross_validation( load_breast_cancer() )