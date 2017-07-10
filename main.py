import  matplotlib.pyplot as plt
import  numpy as np
from sklearn import datasets,linear_model,discriminant_analysis,cross_validation

def load_data():
    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data,diabetes.target)

def test_LinearRegression(*data):
    X_train,X_test,y_train,y_test = data
    regr = linear_model.LinearRegression(normalize=True,n_jobs=4)
    regr.fit(X_train,y_train)
    print('Coefficient:%s,intercept %.2f'%(regr.coef_,regr.intercept_))
    print('Score:%.2f'%regr.score(X_test,y_test))

def test_Ridge(*data):
    X_train,X_test,y_train,y_test = data
    regr = linear_model.Ridge(solver='sag',alpha=0.6)
    regr.fit(X_train,y_train)
    print('Coefficient:%s,intercept %.2f'%(regr.coef_,regr.intercept_))
    print('Score:%.2f'%regr.score(X_test,y_test))


def test_Ridge_alpha(*data):
    alphas = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
    scores= []
    for i,alpha in enumerate(alphas):
        regr =linear_model.Ridge(alpha=alpha)
        regr.fit(X_train,y_train)
        scores.append(regr.score(X_test,y_test))
    fig = plt.figure()
    print("scores:",scores)
    ax = fig.add_subplot(1,1,1)
    ax.set_xscale('log')
    ax.plot(alphas, scores)
    #plt.show()

def test_Elastic_alpaha_rho(*data):
    X_train, X_test, y_train, y_test = data
    alphas = np.logspace(-2,2)
    rhos = np.linspace(0.01,1)
    scores = []
    for alpha in alphas:
        for rho in rhos:
            regr = linear_model.ElasticNet(alpha=alpha,l1_ratio=rho)
            regr.fit(X_train,y_train)
            scores.append(regr.score(X_test,y_test))
            print("alpha=%f,rho=%f"%(alpha,rho))

    alphas,rhos = np.meshgrid(alphas,rhos)
    scores=np.array(scores).reshape(alphas.shape)
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(alphas,rhos,scores,rstride=1,cstride=1,cmap=cm.jet,linewidth=0,antialiased=False)
    fig.colorbar(surf,shrink = 0.5,aspect=5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\rho$")
    ax.set_zlabel("score")
    ax.set_title("ElasticNet")
    plt.show()

if __name__== "__main__":
    X_train, X_test, y_train, y_test = load_data()
    test_LinearRegression(X_train,X_test,y_train,y_test)
    print("----------------------")
    test_Ridge(X_train, X_test, y_train, y_test)
    print("----------------------")
    test_Ridge_alpha(X_train, X_test, y_train, y_test)
    print("----------------------")
    test_Elastic_alpaha_rho(X_train, X_test, y_train, y_test)