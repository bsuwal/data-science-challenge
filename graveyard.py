from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csgraph

def build_weight_matrix(coordinates, k=7):
    """
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    """
    # calculate euclidian distance matrix
    sigma=.05;
    dists = squareform(pdist(coordinates)) 
    squared_dists = np.square(dists)

    # need to find local sigma
    # im only going to pick the kth nearest neighbor
    closest = np.argsort(dists, axis=1)[:, k]

    # construct the local sigma matrix
    local_sigma = np.empty((len(dists), len(dists)))
    for i in range(len(dists)) :
        row = np.full((1, len(dists)), dists[i][closest[i]])    
        local_sigma[i] = row

    W = np.exp(np.negative(np.divide(squared_dists, local_sigma)))
    return W



# do the spectral clustering
weight_matrix = build_weight_matrix(rms_array)
laplacian = csgraph.laplacian(weight_matrix, normed=False)
eigval, eigvec = np.linalg.eig(laplacian)

plt.scatter(range(20), eigval[:20])
plt.show()

##############################################################
##############################################################
##############################################################

# ARIMA code
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(rms['rpm'], order=(20,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
