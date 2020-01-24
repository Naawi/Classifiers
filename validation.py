from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from statistics import stdev
import matplotlib.pyplot as plt


class Validate( object ):

    def __init__( self, data_set ):
        self.data_set = data_set
        self.tree = None
        self.X_test = None
        self.y_test = None
        self.matrix = { 'TP' : 0, 'FP' : 0, 'TN' : 0, 'FN' : 0 }

    def train_model( self, classifier, split ):
        X_train, X_test, y_train, y_test = train_test_split( self.data_set.data, self.data_set.target, 
                                                            test_size = split, random_state = 2 ) # TODO: decide on a value for random_state
        self.tree = classifier.fit( X_train, y_train ) 
        self.X_test = X_test
        self.y_test = y_test

    def get_precision( self ):
        return self.matrix[ 'TP' ] / ( self.matrix[ 'TP' ] + self.matrix[ 'FP' ] )

    def get_recall( self ):
        return self.matrix[ 'TP' ] / ( self.matrix[ 'TP' ] + self.matrix[ 'FN' ] )

    def get_F1_score( self ):
        p = self.get_precision()
        r = self.get_recall()
        return 2*( p*r / (p + r ) )

    def get_accuracy( self ):
        accuracy = self.tree.score( self.X_test, self.y_test )
        return accuracy

    def cross_validate( self, split, runs = 10 ):
        scores = []
        precisions = []
        recalls = []
        for _ in range( runs ):
            self.train_model( tree.DecisionTreeClassifier(criterion="entropy"), split ) # KNeighborsClassifier( n_neighbors = 5 )
            self.generate_confusion_matrix()
            scores.append( self.get_accuracy() )
            precisions.append( self.get_precision() )
            recalls.append( self.get_recall() )

        acc_mean = sum( scores ) / runs
        acc_stdev = stdev( scores )
        print("Accuracy mean: ", acc_mean )
        print("Accuracy stdev: ", acc_stdev )

        prec_mean = sum( precisions ) / runs
        prec_stdev = stdev( precisions )
        print("Precision mean: ", prec_mean )
        print("Precision stdev: ", prec_stdev )

        rec_mean = sum( recalls ) / runs
        rec_stdev = stdev( recalls )
        print("Recall mean: ", rec_mean )
        print("Recall stdev: ", rec_stdev )

    def generate_confusion_matrix( self ):
        for i in range( len( self.X_test ) ):
            prediction = self.tree.predict( [ self.X_test[ i ] ] )
            answer = self.y_test[ i ]

            if prediction == answer:
                if answer: # is true
                    self.matrix[ 'TP' ] += 1
                else:
                    self.matrix[ 'FN' ] += 1
            else:
                if answer: # is true
                    self.matrix[ 'TN' ] += 1
                else:
                    self.matrix[ 'FP' ] += 1
        
        return self.matrix

    def compute_ROC_curve( self, split ):
        X_train, X_test, y_train, y_test = train_test_split( self.data_set.data, self.data_set.target, 
                                                            test_size = split, random_state = 2 ) 
        baseline_probs = [ 0 for _ in range( len( y_test ) ) ]
        model = KNeighborsClassifier( n_neighbors = 5 ).fit( X_train, y_train )

        # predict probabilities. keep positive outcomes only
        lr_probs = model.predict_proba( X_test )[ :, 1 ]
        # calculate roc curves
        baseline_fpr, baseline_tpr, _ = roc_curve( y_test, baseline_probs ) #false positive rate and true positive rate
        model_fpr, model_tpr, _ = roc_curve( y_test, lr_probs )
        # plot the roc curve for the model
        plt.plot( ns_fpr, ns_tpr, linestyle = "--", label = "baseline" )
        plt.plot( lr_fpr, lr_tpr, marker = ".", label = "ROC curve" )
        # axis labels
        plt.xlabel( 'False Positive Rate' )
        plt.ylabel( 'True Positive Rate' )

        plt.legend()
        plt.show()

# Cross validation using sklearn
def cross_validation( data_set ):
    dtree = tree.DecisionTreeClassifier( criterion = "entropy" ).fit( data_set.data, data_set.target )
    cv_scores = cross_val_score( dtree, data_set.data, 
                                 data_set.target, cv = 10 )
    print( "mean:", cv_scores.mean() )
    print( "stdev:", cv_scores.std() )
    
    return dtree



if __name__ == "__main__":
    val = Validate( load_breast_cancer() )
    # val.train_model(0.1)
    val.cross_validate(0.1)
    # val.compute_ROC_curve(0.1)

    # print( "using provided cross validation module:" )
    # cross_validation( load_breast_cancer() )