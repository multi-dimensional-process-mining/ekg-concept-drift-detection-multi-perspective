import ruptures as rpt
import matplotlib.pylab as plt


def rpt_pelt(series, pen=3):
    '''
    From Adams et al. (2021) https://github.com/niklasadams/explainable_concept_drift_pm
    Applies the PELT-algorithm with the provided penalty
    args:
        series: (Reduced) time series, retrieved when applying dimensionality reduction
        pen: penalty value for classifying change points
    returns:
        list of change points
    '''
    algo = rpt.Pelt(model="rbf", min_size=1, jump=1).fit(series)
    result = algo.predict(pen=pen)
    # display
    # rpt.display(series, result)
    # plt.show()
    return result[:-1]
