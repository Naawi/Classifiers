from sklearn.datasets import make_regression, load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from random import randint

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


    def update_model_batch( self, input, output ):
        # update w0
        diffs = []
        for i in range( len( input ) ):
            pred = self.predict( input[ i ] )
            diff = output[ i ] - pred
            diffs.append( diff )
        self.weights[ 0 ] += self.alpha*sum( diffs )

        # update other weights
        for i in range( 1, len( self.weights ) ): 
            # work out diff*xi
            diff_sum = 0
            for j in range( len( diffs ) ):
                diff_sum += diffs[ i ]*input[ j ][ i - 1 ] # multiply it with the right feature i
            
            self.weights[ i ] += self.alpha*diff_sum


    def learn( self, data, answers, show_learning_curve = False, stochastic = True ):
        if stochastic: # random data points are selected to avoid biasing the classifier 
            errors = []
            for i in range( len( data )*2 ): # perform update 2n times
                rnd = randint( 0, len( data ) - 1 )
                error = self.update_model_stochastic( data[ rnd ], answers[ rnd ] )
                errors.append( error )

            if show_learning_curve:
                self._plot_learning_curve( errors )
        else:
            for i in range( len( data ) ):
                self.update_model_batch( data, answers )


    def get_max_error( self, input, output ):
        errs = []
        for i in range( len( input ) ):
            errs.append( output[ i ] - self.predict( input[ i ] ) )

        return max( errs )


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
    num_features = 1
    gd = GradientDescent(num_features)
    X, y = make_regression( n_samples = 100, n_features = num_features, noise = 2 )
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, 
                                                               random_state = 0 )
    gd.learn( X_train, y_train, stochastic = False )
    #print (gd.weights)
    gd.test( X_test, y_test )

