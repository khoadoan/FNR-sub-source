# import torch
# from torch import nn

# from deepctr_torch.models.basemodel import BaseModel
# from deepctr_torch.inputs import combined_dnn_input
# from deepctr_torch.layers import FM, DNN
# import torch.utils.data as Data
# from torch.utils.data import DataLoader

# import pandas as pd
# import torch
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# from sklearn.preprocessing import LabelEncoder
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences

# from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
# from deepctr_torch.models import DeepFM

# class MyDeepFM(BaseModel):
#     """Instantiates the DeepFM Network architecture.
#     :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
#     :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
#     :param use_fm: bool,use FM part or not
#     :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
#     :param l2_reg_linear: float. L2 regularizer strength applied to linear part
#     :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
#     :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
#     :param init_std: float,to use as the initialize std of embedding vector
#     :param seed: integer ,to use as random seed.
#     :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
#     :param dnn_activation: Activation function to use in DNN
#     :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
#     :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
#     :param device: str, ``"cpu"`` or ``"cuda:0"``
#     :return: A PyTorch model instance.
    
#     """

#     def __init__(self,
#                  linear_feature_columns, dnn_feature_columns, use_fm=True,
#                  dnn_hidden_units=(256, 128),
#                  l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
#                  dnn_dropout=0,
#                  dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', lr=0.01, use_linear_model=True):

#         super().__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
#                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
#                                      device=device)

#         self.use_fm = use_fm
#         self.use_dnn = len(dnn_feature_columns) > 0 and len(
#             dnn_hidden_units) > 0
#         if use_fm:
#             self.fm = FM()

#         if self.use_dnn:
#             self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
#                            activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
#                            init_std=init_std, device=device)
#             self.dnn_linear = nn.Linear(
#                 dnn_hidden_units[-1], 1, bias=False).to(device)

#             self.add_regularization_weight(
#                 filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
#             self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
#         self.to(device)
        
#         self.lr = lr
#         self.use_linear_model = use_linear_model

#     def forward(self, X):

#         sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
#                                                                                   self.embedding_dict)
        
#         if self.use_linear_model:
#             #for x in sparse_embedding_list:
#             #    print(x.shape)
#             #return sparse_embedding_list, dense_value_list
#             logit = self.linear_model(X)
#         else:
#             logit = 0.0

#         if self.use_fm and len(sparse_embedding_list) > 0:
#             fm_input = torch.cat(sparse_embedding_list, dim=1)
#             logit += self.fm(fm_input)

#         if self.use_dnn:
#             dnn_input = combined_dnn_input(
#                 sparse_embedding_list, dense_value_list)
#             dnn_output = self.dnn(dnn_input)
#             dnn_logit = self.dnn_linear(dnn_output)
#             logit += dnn_logit

#         y_pred = self.out(logit)

#         return y_pred
    
#     def _get_optim(self, optimizer):
#         if isinstance(optimizer, str):
#             print('Training using optimizer {} with lr {}'.format(optimizer, self.lr))
#             if optimizer == "sgd":
#                 optim = torch.optim.SGD(self.parameters(), lr=self.lr)
#             elif optimizer == "adam":
#                 optim = torch.optim.Adam(self.parameters(), lr=self.lr)  # 0.001
#             elif optimizer == "adagrad":
#                 optim = torch.optim.Adagrad(self.parameters(), self.lr)  # 0.01
#             elif optimizer == "rmsprop":
#                 optim = torch.optim.RMSprop(self.parameters(), self.lr)
#             else:
#                 raise NotImplementedError
#         else:
#             optim = optimizer
#         return optim
    
#     def _get_input_features(self, X):
#         sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
#                                                                                   self.embedding_dict)
#         return sparse_embedding_list, dense_value_list
    
#     def get_input_features(self, x, user_columns, item_columns, batch_size=256):
#         """
#         :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
#         :param batch_size: Integer. If unspecified, it will default to 256.
#         :return: Numpy array(s) of predictions.
#         """
#         model = self.eval()
#         if isinstance(x, dict):
#             x = [x[feature] for feature in self.feature_index]
#         for i in range(len(x)):
#             if len(x[i].shape) == 1:
#                 x[i] = np.expand_dims(x[i], axis=1)

#         tensor_data = Data.TensorDataset(
#             torch.from_numpy(np.concatenate(x, axis=-1)))
#         test_loader = DataLoader(
#             dataset=tensor_data, shuffle=False, batch_size=batch_size)

#         user_features, item_features = [], []
#         with torch.no_grad():
#             for index, x_test in enumerate(test_loader):
#                 x = x_test[0].to(self.device).float()

#                 sparse_embedding_list, dense_value_list = self._get_input_features(x)
                
#                 batch_user_features = torch.cat([sparse_embedding_list[i].squeeze() for i in user_columns], dim=1).cpu().data.numpy()
#                 batch_item_features = torch.cat([sparse_embedding_list[i].squeeze() for i in item_columns], dim=1).cpu().data.numpy()
                
                
#                 user_features.append(batch_user_features)
#                 item_features.append(batch_item_features)
#         return np.vstack(user_features), np.vstack(item_features)
    
# class MyDeepFMSimple(MyDeepFM):
#     def __init__(self,
#                  linear_feature_columns, dnn_feature_columns, use_fm=True,
#                  dnn_hidden_units=(256, 128),
#                  l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
#                  dnn_dropout=0,
#                  dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', lr=0.01, use_linear_model=True):

#         super().__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
#                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
#                                      device=device)

#         self.use_fm = use_fm
#         self.use_dnn = len(dnn_feature_columns) > 0 and len(
#             dnn_hidden_units) > 0
#         if use_fm:
#             self.fm = FM()

#         if self.use_dnn:
#             self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
#                            activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
#                            init_std=init_std, device=device)
#             self.dnn_linear = nn.Linear(
#                 dnn_hidden_units[-1], 1, bias=False).to(device)

#             self.add_regularization_weight(
#                 filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
#             self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
#         self.to(device)
        
#         self.lr = lr
#         self.use_linear_model = use_linear_model

#     def forward(self, user_features, item_features):
#         bsize = user_features.size(0)
#         dense_value_list = []
#         sparse_embedding_list = [user_features.view(bsize, 1, self.dnn_feature_columns[0].embedding_dim)]
#         sparse_embedding_list += [f.view(f.size(0), 1, f.size(1)) for f in item_features.split([f.embedding_dim for f in self.dnn_feature_columns[1:]], dim=1)]
        
#         if self.use_linear_model:
#             #for x in sparse_embedding_list:
#             #    print(x.shape)
#             #return sparse_embedding_list, dense_value_list
#             logit = self.linear_model(X)
#         else:
#             logit = 0.0

#         if self.use_fm and len(sparse_embedding_list) > 0:
#             fm_input = torch.cat(sparse_embedding_list, dim=1)
#             logit += self.fm(fm_input)

#         if self.use_dnn:
#             dnn_input = combined_dnn_input(
#                 sparse_embedding_list, dense_value_list)
#             dnn_output = self.dnn(dnn_input)
#             dnn_logit = self.dnn_linear(dnn_output)
#             logit += dnn_logit

#         y_pred = self.out(logit)

#         return y_pred 
    
#     def predict(self, user_features, item_features, batch_size=256):
#         """
#         :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
#         :param batch_size: Integer. If unspecified, it will default to 256.
#         :return: Numpy array(s) of predictions.
#         """
#         model = self.eval()
        
#         tensor_data = Data.TensorDataset(
#             torch.from_numpy(user_features), torch.from_numpy(item_features))
#         test_loader = DataLoader(
#             dataset=tensor_data, shuffle=False, batch_size=batch_size)

#         pred_ans = []
#         with torch.no_grad():
#             for index, (x_user, x_item) in enumerate(test_loader):
#                 x_user = x_user.to(self.device)
#                 x_item = x_item.to(self.device)

#                 y_pred = model(x_user, x_item).cpu().data.numpy()  # .squeeze()
#                 pred_ans.append(y_pred)

#         return np.concatenate(pred_ans).astype("float64")
    
    

import torch
from torch import nn

from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import FM, DNN
import torch.utils.data as Data
from torch.utils.data import DataLoader

import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
from deepctr_torch.models.deepfm import MyDeepFM

def create_NeuCFydata_DeepFM(saved_model_path, device='cpu', return_model=False):
    checkpoint = torch.load(saved_model_path)
    dnn_hidden_units = checkpoint['dnn_hidden_units']
    fixlen_feature_columns = checkpoint['fixlen_feature_columns']
    varlen_feature_columns = checkpoint['varlen_feature_columns']
    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    best_model = MyDeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device, dnn_hidden_units=dnn_hidden_units)
    best_model.load_state_dict(checkpoint['model'])
    best_model.eval()
    
    def neural_model_func(u, i):
        return best_model.predict_example(u, i)
    
    if return_model:
        return best_model, neural_model_func
    else:
        return neural_model_func

    