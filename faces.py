import  torch
import  numpy       as np

import  torch.nn    as nn
from    torch.utils.data    import DataLoader
from    torchvision         import transforms, datasets
from    facenet_pytorch     import MTCNN, InceptionResnetV1
from    torchsummary        import summary

from    PIL                 import Image
from    utils               import *
from    visutils            import *
from    joblib              import dump, load

from    sklearn.neighbors   import NearestNeighbors
from    sklearn.svm         import SVC
from    sklearn.manifold    import TSNE

import  umap        as um

import  matplotlib.pyplot as plt
import  matplotlib.colors as mcolors



from    matplotlib.colors import ListedColormap
from    sklearn.model_selection import train_test_split
from    sklearn.preprocessing import StandardScaler
from    sklearn.pipeline import make_pipeline
from    sklearn.datasets import make_moons, make_circles, make_classification
from    sklearn.neural_network import MLPClassifier
from    sklearn.neighbors import KNeighborsClassifier
from    sklearn.svm import SVC
from    sklearn.gaussian_process import GaussianProcessClassifier
from    sklearn.gaussian_process.kernels import RBF
from    sklearn.tree import DecisionTreeClassifier
from    sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from    sklearn.naive_bayes import GaussianNB
from    sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from    sklearn.inspection import DecisionBoundaryDisplay

from    tqdm import tqdm


def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity according 
    to the definition of the dot product"""
    a = a.cpu()
    b = b.cpu()
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def cos(a,b):
    """
    cos_sim returns real numbers,where negative numbers have different interpretations.
    so we use this function to return only positive values.
    """
    minx = -1 
    maxx = 1
    return (cos_sim(a,b)- minx)/(maxx-minx)

def whichDevice():
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    return device

################################################################################ 
################################################################################ 
################################################################################ 
def classifierSel(model_name, **confs):
    models = {
        "svm"   : SVC
        }
    return models[model_name](**confs)

def clusterSel(model_name, **confs):
    models = {
        "nn"    : NearestNeighbors,
        }
    return models[model_name](**confs)


################################################################################
################################################################################
################################################################################
class FaceRecognition:
    def __init__(self):
        self.device = whichDevice()
        self.mtcnn      = MTCNN(    image_size=160, margin=0, min_face_size=20,
                                    thresholds=[0.6, 0.7, 0.7], 
                                    factor=0.709, device= self.device)
        self.model      = InceptionResnetV1(pretrained='vggface2')
        self.model      = self.model.eval().to(self.device) 
        self.emb_path   = 'db/list'

        try:
            
            self.classifier = load(self.emb_path+'/classifier.joblib')
            self.clusters   = load(self.emb_path+'/clusters.joblib') 
            self.targets    = u_fileNumberList2array(self.emb_path+'/targets.txt')
            self.targetsv   = u_fileNumberList2array(self.emb_path+'/targets_val.txt')
            self.names      = u_loadJson(self.emb_path+'/names.txt')
            self.namesv     = u_loadJson(self.emb_path+'/names_val.txt')
            

        except:
            print('First define the embeddings' +
                  'running getDbEmbeddings function')

              

       
    #..........................................................................
    # creating embeddings for the dataset
    def getDbEmbeddings(self):

        ## classifier confs
        data            = u_loadJson('confs.json')
        classifier_data =  data['classifier']
        clustering_data =  data['clustering']

        train_flag      = data['train_flag']

        cluster_name    = data['cluster_name']
        classifier_name = data['classifier_name']
        
        clusters        = clusterSel(cluster_name, **clustering_data[cluster_name])
        classifier      = classifierSel(classifier_name, **classifier_data[classifier_name])


        # remember each folder must coint only pidctures for one person
        dataset = datasets.ImageFolder("db/data/train" if train_flag == 1 else "db/data/val")
        dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
        loader  = DataLoader(dataset, collate_fn=lambda x: x[0])

        aligned = []
        names   = []
        for _, (x, y) in enumerate(tqdm(loader)):
            x_aligned, prob = self.mtcnn(x, return_prob=True)
            if x_aligned is not None:
                x_aligned   = x_aligned.to(self.device)
                x_aligned   = torch.unsqueeze(x_aligned, dim =0)

                emb         = self.model(x_aligned)
                emb         = torch.squeeze(emb)
                aligned.append(emb.cpu().detach())
                

            else:
                plt.imshow(x, interpolation='nearest')
                plt.show()

        
        aligned     = torch.stack(aligned)
        
        emb         = aligned.numpy()

        if train_flag == 1:
            clusters.fit(emb)
            classifier.fit(emb, dataset.targets)

            # saving the neares neighbors model and other files
            dump(clusters, self.emb_path+'/clusters.joblib')
            dump(classifier, self.emb_path+'/classifier.joblib')

            u_saveArray2File(self.emb_path+'/targets.txt', dataset.targets)
            u_saveDict2File(self.emb_path+'/names.txt', dataset.idx_to_class)
            u_saveArrayTuple2File(self.emb_path+'/embs.txt', emb)

        else:
            u_saveArray2File(self.emb_path+'/targets_val.txt', dataset.targets)
            u_saveDict2File(self.emb_path+'/names_val.txt', dataset.idx_to_class)
            u_saveArrayTuple2File(self.emb_path+'/embs_val.txt', emb)
        
    #..........................................................................
    # face recognition 
    # getting the embeddings for the data set, remember the data set must contain
    # names in folders and images inside
    def faceRecog(self, img):
        aligned, prob   = self.mtcnn(img, return_prob=True)
        data            = {'id': -1, 'dist': -1}
       
        if aligned is not None:
            aligned     = aligned.unsqueeze(0).to(self.device)
            embedding   = self.model(aligned)
            embedding   = embedding.cpu().detach().numpy()                

            distances, indices = self.clusters.kneighbors(embedding)
            
            if len(distances) > 0: 
                data['id']  = int(np.int64(self.classifier.predict(embedding)[0]))
                data['dist']    = distances[0][0]   - 0.3

        return [data]  

    #...........................................................................
    # visualization
    # for dataset understanding
    def visualization(self):
        #embs    = np.array(u_fileNumberMat2array(self.emb_path+'/embs_val.txt'))   
        #embs_   = TSNE(n_components=2).fit_transform(embs)
                
        #chart(embs_,  np.array(self.targets))
        #scatter2D(embs_, np.array(self.targets))

        #embs_   = um.UMAP().fit_transform(embs)
        #scatter2D(embs_, np.array(self.targetsv))

        #summary(self.model, (3,160,160))
        #print(torch.cuda.memory_summary())
        pass
                


    #...........................................................................
    def validation(self):
        names = [
            "Nearest Neighbors",
            "Linear SVM",
            "RBF SVM",
            "Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net",
            "AdaBoost",
            "Naive Bayes",
            "QDA",
        ]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
        ]


        X   = np.array(u_fileNumberMat2array(self.emb_path+'/embs.txt'))
        X_  = TSNE(n_components=2).fit_transform(X)
        y   = self.targets


        Xt  = np.array(u_fileNumberMat2array(self.emb_path+'/embs_val.txt'))
        Xt_ = TSNE(n_components=2, perplexity=20).fit_transform(Xt)
        yt  = self.targetsv

        X_train, X_test, y_train, y_test = X, Xt, y, yt 


        x_min, x_max = Xt_[:, 0].min() - 0.5, Xt_[:, 0].max() + 0.5
        y_min, y_max = Xt_[:, 1].min() - 0.5, Xt_[:, 1].max() + 0.5

            
        figure = plt.figure(figsize=(3, 3))
        i = 1
        ax = plt.subplot(1, 11, i)


        
        keys = {v: k for k, v in self.names.items()}
        
        cm1 = colorMapByNumber(6)

        for it, name in enumerate(yt):
            yt[it] = int(keys[self.namesv[str(int(name))]])

        # Plot the testing points
        ax.scatter(Xt_[:, 0], Xt_[:, 1], c=y_test, cmap=cm1)
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        

        
        # iterate over classifiers
        for name, clf in zip(names, classifiers):

            ax = plt.subplot(1, 11, i)

            clf = make_pipeline(StandardScaler(), clf)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, yt)

            pred    = clf.predict(X_test)
            ok      = (pred == yt).nonzero()
            
            
            ax.scatter(Xt_[:, 0], Xt_[:, 1], c='black', cmap=cm1, alpha=0.3, edgecolors="k")
            ax.scatter(Xt_[ok, 0], Xt_[ok, 1], c='blue', cmap=cm1)
            
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)


            ax.text(
                x_max - 0.35,
                y_min + 0.35,
                ("%.2f" % score).lstrip("0"),
                size=10,
                horizontalalignment="right",
        )


            i += 1
        
        plt.show()
            
                
    ############################################################################
    ############################################################################
    ############################################################################
    #...........................................................................
    def validation_(self):
        names = [
            "Nearest Neighbors",
            "Linear SVM",
            "RBF SVM",
            "Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net",
            "AdaBoost",
            "Naive Bayes",
            "QDA",
        ]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
        ]


        X   = np.array(u_fileNumberMat2array(self.emb_path+'/embs.txt'))
        X_  = TSNE(n_components=2).fit_transform(X)
        y   = self.targets


        Xt  = np.array(u_fileNumberMat2array(self.emb_path+'/embs_val.txt'))
        Xt_ = TSNE(n_components=2, perplexity=20).fit_transform(Xt)
        yt  = self.targetsv

        X_train, X_test, y_train, y_test = X, Xt, y, np.array(yt).astype(int)

        grad = [-1]
        for it in range(1, len(yt)):
            if yt[it] != y_test[it-1]:
                grad.append(it-1)
        
        grad.append(len(yt)-1)

        # this var is for plot position
        i = 1
        
        # initializing the plot
        fig = make_subplots(
            rows=1, cols=11,
            subplot_titles=np.concatenate((['Points'] , names)),
            shared_yaxes=True,
            #row_heights = [0.95, 0.05]
            )

        # ploting by person

        names_val  = u_loadJson(self.emb_path+'/names_val.txt')

        for it in range(len(grad)-1):
            ini = grad[it]+1
            fin = grad[it+1]

            fig.add_trace(go.Scatter(  x = Xt_[ini:fin+1, 0], y=Xt_[ini:fin+1, 1], mode='markers',
                                    marker_color= it*10 , name = names_val[str(y_test[ini])]
                                 ), 
                      row=1, 
                      col=i ) 
        
        keys = {v: k for k, v in self.names.items()}
        
        for it, name in enumerate(yt):
            yt[it] = int(keys[self.namesv[str(int(name))]])

        i += 1


        # iterate over classifiers
        for name, clf in zip(names, classifiers):

            clf = make_pipeline(StandardScaler(), clf)
            clf.fit(X_train, y_train)
            score = round(clf.score(X_test, yt),4)
            
            pred    = clf.predict(X_test)
            ok      = (pred == yt).nonzero()
            
           
            fig.add_trace(
                        go.Scatter( x = Xt_[:, 0], y=Xt_[:, 1], mode='markers',
                                    marker_color= 'gray', showlegend=False
                                 ), 
                        row=1, 
                        col=i)
            
            rx = Xt_[ok, 0].reshape(-1)
            ry = Xt_[ok, 1].reshape(-1) 

            fig.add_trace(
                        go.Scatter( x = rx, y=ry, mode='markers',
                                    marker_color='green', showlegend=False
                                 ), 
                        row=1, 
                        col=i ) 

            fig.update_xaxes(title_text="Score: "+str(score), row=1, col=i)

            i   += 1
          
        ###..........  
                
        #fig.update(layout_showlegend=False)        
        fig.show()
        

        
