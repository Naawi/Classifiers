from sklearn.datasets import make_regression, load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class GradientDescent( object ):

    def __init__( self, num_features = 1 ):
        self.weights = [ 0 for _ in range ( num_features + 1 ) ] # linear gradient descent is default
        self.alpha = 0.1 # learning rate

    def predict( self, input ):
        pred = self.weights[ 0 ]
        for i in range( len( input ) ):
            pred += self.weights[ i + 1 ]*input[ i ] 
        return pred

    def update_model_stochastic( self, input, output ):
        pred = self.predict( input )
        error = output - pred

        self.weights[ 0 ] += self.alpha*error
        for i in range( len( input ) ):
            self.weights[ i + 1 ] += self.alpha*error*input[ i ]

        return abs( error )

    def update_model_batch( self, errors, input, output ):
        self.weights[ 0 ] += self.alpha*sum( errors )

        sum = 0
        for j in range( errors ):
            sum += errors [ j ]*input[ j ]

        for i in range( 1, len( self.weights ) ): # check correctness. how does this work with more than 2 weights (i.e. > 1 feature)
            self.weights[ i ] += self.alpha*sum

    def learn( self, data, answers, show_learning_curve = False, stochastic = True ):
        errors = []
        for i in range( len( data ) ):
            if stochastic:
                error = self.update_model_stochastic( data[ i ], answers[ i ] )
            else:
                pred = self.predict( input )
                error = output - pred   
            errors.append( error )
        
        if not stochastic:
            self.update_model_batch( errors, data, answers )

        if show_learning_curve:
            self._plot_learning_curve( errors )

    def _plot_learning_curve( self, errors ):
        pos_errs = [ abs( err ) for err in errors ]
        plt.plot( range( len( errors ) ), pos_errs, linewidth=0.6 )
        plt.xlabel( 'Data points' )
        plt.ylabel( 'Error rate' )
        plt.ylim( bottom = 0 )
        plt.xlim( 0 )
        plt.show()

    def test( self, data, answers, show_curve = True ):
        squared_diffs = 0
        predictions = []
        for i in range( len( data ) ):
            pred = self.predict( data[ i ] )
            squared_diffs += abs( answers[ i ] - pred )**2
            predictions.append( pred )

        mean = squared_diffs / len( data )
        print( "Mean squared error:", mean )

        if show_curve:
            plt.scatter( data, answers, color="red" )
            plt.scatter( data, predictions, color="blue" )
            plt.show()



if __name__ == "__main__":
    gd = GradientDescent()
    X, y = make_regression( n_samples = 100, n_features = 1, noise = 2 )
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, 
                                                               random_state = 0 )
    gd.learn( X_train, y_train )
    gd.test( X_test, y_test, True )


