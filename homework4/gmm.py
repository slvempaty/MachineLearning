import json
import random
import numpy as np

def multivariate_guassian(point,mu,cov):
    xMinmu=np.subtract(point,mu)
    inv_cov = np.linalg.inv(cov)
    expo=np.dot(np.dot(xMinmu.T,inv_cov),xMinmu)
    numa = np.exp(-0.5*expo)
    deno = np.sqrt( np.linalg.det(cov))
    guassianVal = numa/deno

    return guassianVal

def gmm_clustering(X, K):
    """
    Train GMM with EM for clustering.
    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - K: A int, the number of total cluster centers
    Returns:
    - mu: A list of all K means in GMM, each elements is a list of 2
    - cov: A list of all K covariance in GMM, each elements is a list of 4
            (note that covariance matrix is symmetric)
    """

    # Initialization:
    pi = []
    mu = []
    cov = []
    for k in range(K):
        pi.append(1.0 / K)
        mu.append(list(np.random.normal(0, 0.5, 2)))
        temp_cov = np.random.normal(0, 0.5, (2, 2))
        temp_cov = np.matmul(temp_cov, np.transpose(temp_cov))
        cov.append(list(temp_cov.reshape(4)))
        #cov.append(list(temp_cov))

    ### you need to fill in your solution starting here ###
    
    # Run 100 iterations of EM updates
    for k in range(K):
        cov[k]=list(np.array(cov[k]).reshape((2,2)))


    X_arr=np.array(X)
    mu_arr=np.array(mu)
    cov_arr=np.array(cov)
    pi_arr=np.array(pi)
    pzk_arr=np.zeros((len(X),K))

   

    num=len(X )

    for t in range(100):
        # p_kn computation #

       # print(pi_arr)

        for i in range(len(X)):
            deno=0

            for k in range(K):
                deno+=(multivariate_guassian(X[i],mu_arr[k],cov_arr[k])*pi_arr[k])
            for j in range(K):
                pzk_arr[i][j]=(multivariate_guassian(X[i],mu_arr[j],cov_arr[j])*pi_arr[j])/deno


        #pi,mu,gamma computation#

        
        for j in range(K):
            tempcount_k = 0
            temp_mu_k=np.zeros(mu_arr[0].shape)
            temp_cov_k=np.zeros(cov_arr[0].shape)
            
            for i in range(num):
                tempcount_k+=pzk_arr[i][j]

            #print(j,tempcount_k/num)

            pi_arr[j]= tempcount_k/num
         
            
            for i in range(num):
                temp_mu_k += (pzk_arr[i][j] * X_arr[i])

            mu_arr[j] = temp_mu_k / tempcount_k


            for i in range(num):
                xMinmu=np.subtract(X_arr[i],mu_arr[j])
                xMinmu=xMinmu[:,np.newaxis]
                #print(xMinmu.shape,xMinmu.T.shape)
                temp_cov_k += (pzk_arr[i][j] * np.dot(xMinmu,np.transpose(xMinmu)))

            cov_arr[j]=temp_cov_k / tempcount_k


    cov=list()
    mu=mu_arr.tolist()
    for i in range(K):
        cov.append(list(cov_arr[i].reshape(4)))

    return mu, cov


def main():
    # load data
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    cov_all = {}

    print('GMM clustering')
    for i in range(5):
        np.random.seed(i)
        mu, cov = gmm_clustering(data_blob, K=3)
        mu_all[i] = mu
        cov_all[i] = cov

        print('\nrun' + str(i) + ':')
        print('mean')
        print(np.array_str(np.array(mu), precision=4))
        print('\ncov')
        print(np.array_str(np.array(cov), precision=4))

    with open('gmm.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'cov': cov_all}, f_json)


if __name__ == "__main__":
    main()
