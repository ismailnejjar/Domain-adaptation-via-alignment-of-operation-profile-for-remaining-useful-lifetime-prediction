import torch
import torch.nn as nn
from source.utils import ReverseLayerF

# --- Helper for standard CNN Feature Extractor ---
def build_feature_extractor(num_features, n_ch, n_k):
    """
    Builds the standard 3-layer CNN feature extractor used across models.
    Preserves original naming for state_dict compatibility.
    """
    layers = nn.Sequential()
    
    # 1st Layer
    layers.add_module('conv1', nn.Conv1d(in_channels=num_features, out_channels=n_ch, 
                                       kernel_size=n_k, stride=1, padding='same'))
    layers.add_module('relu1', nn.ReLU(True))
    
    # 2nd Layer
    layers.add_module('conv2', nn.Conv1d(in_channels=n_ch, out_channels=n_ch, 
                                       kernel_size=n_k, stride=1, padding='same'))
    layers.add_module('relu2', nn.ReLU(True))
    
    # 3rd Layer
    layers.add_module('conv3', nn.Conv1d(in_channels=n_ch, out_channels=1, 
                                       kernel_size=n_k, stride=1, padding='same'))
    layers.add_module('relu3', nn.ReLU(True))
    
    return layers

# --- Helper for RUL Regressor ---
def build_rul_regressor(length, n_neurons):
    """
    Builds the standard RUL regression head.
    """
    layers = nn.Sequential()
    layers.add_module('fc_1', nn.Linear(in_features=length, out_features=n_neurons))
    layers.add_module('relu4', nn.ReLU(True))
    layers.add_module('fc_2', nn.Linear(in_features=n_neurons, out_features=1))
    layers.add_module('sigmoid', nn.Sigmoid())
    return layers

# --- Helper for Domain Classifier ---
def build_domain_classifier(length, output_dim=1):
    """
    Builds a standard domain classifier head.
    """
    layers = nn.Sequential()
    layers.add_module('dc_fc1', nn.Linear(in_features=length, out_features=50))
    layers.add_module('dc_relu1', nn.ReLU(True))
    layers.add_module('dc_fc2', nn.Linear(in_features=50, out_features=30))
    layers.add_module('dc_relu2', nn.ReLU(True))
    layers.add_module('dc_fc3', nn.Linear(in_features=30, out_features=output_dim))
    
    if output_dim == 1:
        layers.add_module('dc_sigmoid', nn.Sigmoid())
    return layers


class CNN_DANN_OCS_hard(nn.Module):
    '''
    OCS-DANN (hard)
    Uses 3 separate binary domain classifiers for Ascending, Steady, and Descending modes.
    '''
    def __init__(self, num_features=18, length=50, n_ch=10, n_k=10, n_neurons=50):
        super().__init__()
        
        # Encoder
        self.feature = build_feature_extractor(num_features, n_ch, n_k)
        
        # Domain Classifiers (One per Operating Condition)
        self.domain_classifier1 = build_domain_classifier(length) # Ascending
        self.domain_classifier2 = build_domain_classifier(length) # Steady
        self.domain_classifier3 = build_domain_classifier(length) # Descending
        
        # RUL Regressor
        self.rul_regressor = build_rul_regressor(length, n_neurons)
        
    def forward(self, x, align_int=1, return_embedding=False):
        # Permute: (bs, length, features) -> (bs, features, length)
        x = x.permute(0, 2, 1)
        
        # Feature Extraction
        feature = self.feature(x)
        feature = torch.flatten(feature, start_dim=1)
        
        # RUL Prediction
        rul_output = self.rul_regressor(feature)
        
        if return_embedding:
            return rul_output.squeeze(1), feature

        # Gradient Reversal
        reverse_feature = ReverseLayerF.apply(feature, align_int)
        
        # Domain Classification
        domain_output1 = self.domain_classifier1(reverse_feature)
        domain_output2 = self.domain_classifier2(reverse_feature)
        domain_output3 = self.domain_classifier3(reverse_feature)

        return rul_output.squeeze(1), [domain_output1.squeeze(1), 
                                       domain_output2.squeeze(1), 
                                       domain_output3.squeeze(1)]


class CNN_DANN_OCS(nn.Module):
    '''
    OCS-DANN (soft)
    Includes an Operating Condition (OC) Classifier to weight the domain losses.
    '''
    def __init__(self, num_features=18, length=50, n_ch=10, n_k=10, n_neurons=50):
        super().__init__()
        
        # Encoder
        self.feature = build_feature_extractor(num_features, n_ch, n_k)
        
        # Domain Classifiers
        self.domain_classifier1 = build_domain_classifier(length)
        self.domain_classifier2 = build_domain_classifier(length)
        self.domain_classifier3 = build_domain_classifier(length)
        
        # RUL Regressor
        self.rul_regressor = build_rul_regressor(length, n_neurons)
        
        # OC Classifier
        self.oc_classifier = nn.Sequential()
        self.oc_classifier.add_module('oc_fc1', nn.Linear(in_features=length, out_features=50))
        self.oc_classifier.add_module('oc_relu1', nn.ReLU(True))
        self.oc_classifier.add_module('oc_fc2', nn.Linear(in_features=50, out_features=30))
        self.oc_classifier.add_module('oc_relu2', nn.ReLU(True))
        self.oc_classifier.add_module('oc_fc3', nn.Linear(in_features=30, out_features=3))
        # LogSoftmax/Softmax usually applied in loss function or training loop
        
    def forward(self, x, align_int=1, return_embedding=False):
        x = x.permute(0, 2, 1)
        
        feature = self.feature(x)
        feature = torch.flatten(feature, start_dim=1)
        
        rul_output = self.rul_regressor(feature)
        
        # OC Prediction
        oc_output = self.oc_classifier(feature)
        
        if return_embedding:
            return rul_output.squeeze(1), feature
        
        # Gradient Reversal
        reverse_feature = ReverseLayerF.apply(feature, align_int)
        
        # Domain Classification
        domain_output1 = self.domain_classifier1(reverse_feature)
        domain_output2 = self.domain_classifier2(reverse_feature)
        domain_output3 = self.domain_classifier3(reverse_feature)

        return rul_output.squeeze(1), [domain_output1.squeeze(1), 
                                       domain_output2.squeeze(1), 
                                       domain_output3.squeeze(1)], oc_output


class CNN_1D_DANN(nn.Module):
    '''
    Standard DANN (Binary Domain Adaptation)
    '''
    def __init__(self, num_features=18, length=50, n_ch=10, n_k=10, n_neurons=50):
        super().__init__()
        
        self.feature = build_feature_extractor(num_features, n_ch, n_k)
        self.domain_classifier = build_domain_classifier(length)
        self.rul_regressor = build_rul_regressor(length, n_neurons)
        
    def forward(self, x, align_int=1, return_embedding=False):
        x = x.permute(0, 2, 1)
        
        feature = self.feature(x)
        feature = torch.flatten(feature, start_dim=1)
        
        rul_output = self.rul_regressor(feature)

        if return_embedding:
            return rul_output.squeeze(1), feature
        
        reverse_feature = ReverseLayerF.apply(feature, align_int)
        domain_output = self.domain_classifier(reverse_feature)
        
        return rul_output.squeeze(1), domain_output.squeeze(1)


class CNN_1D(nn.Module):
    '''
    Baseline 1D CNN (Source Only)
    '''
    def __init__(self, num_features=18, length=50, n_ch=10, n_k=10, n_neurons=50):
        super().__init__()
        
        self.feature = build_feature_extractor(num_features, n_ch, n_k)
        self.rul_regressor = build_rul_regressor(length, n_neurons)
        
    def forward(self, x, return_embedding=False):
        x = x.permute(0, 2, 1)
        
        feature = self.feature(x)
        feature = torch.flatten(feature, start_dim=1)
        
        output = self.rul_regressor(feature)
        
        if return_embedding:
            return output.squeeze(1), feature
        
        return output.squeeze(1)


class CNN_1D_MMD(nn.Module):
    '''
    MK-MMD baseline model
    Structurally similar to CNN_1D but often used with MMD loss on the feature vector.
    '''
    def __init__(self, num_features=18, length=50, n_ch=10, n_k=10, n_neurons=50):
        super().__init__()
        
        self.feature = build_feature_extractor(num_features, n_ch, n_k)
        self.rul_regressor = build_rul_regressor(length, n_neurons)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        feature = self.feature(x)
        feature = torch.flatten(feature, start_dim=1)
        
        output = self.rul_regressor(feature)
        
        # MMD models typically return features for loss calculation
        return output.squeeze(1), feature


class CNN_Altitude_Regressor(nn.Module):
    '''
    OOD-Regressor: Regresses RUL and classifies Altitude/Outlier score.
    '''
    def __init__(self, num_features=18, length=50, n_ch=10, n_k=10, n_neurons=50):
        super().__init__()
        
        self.feature = build_feature_extractor(num_features, n_ch, n_k)
        
        # Altitude Classifier (Similar structure to domain classifier)
        self.altitude_classifier = build_domain_classifier(length)
        
        self.rul_regressor = build_rul_regressor(length, n_neurons)
        
    def forward(self, x, align_int=1):
        x = x.permute(0, 2, 1)
        
        feature = self.feature(x)
        feature = torch.flatten(feature, start_dim=1)
        
        # Reverse Layer often used here if adversarial training on altitude is desired
        reverse_feature = ReverseLayerF.apply(feature, align_int)
        
        altitude_output = self.altitude_classifier(reverse_feature)
        rul_output = self.rul_regressor(feature)
        
        return rul_output.squeeze(1), altitude_output.squeeze(1)


class Multilayer_LSTM(nn.Module):
    '''
    Standard LSTM for RUL estimation.
    '''
    def __init__(self, num_features=18, hidden_units=128, num_layers=1, n_neurons=32):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=num_layers
        )
        
        self.rul_regressor = nn.Sequential()
        self.rul_regressor.add_module('fc1', nn.Linear(in_features=hidden_units, out_features=n_neurons))
        self.rul_regressor.add_module('relu1', nn.ReLU(True))
        self.rul_regressor.add_module('fc2', nn.Linear(in_features=n_neurons, out_features=1))
        self.rul_regressor.add_module('sigmoid', nn.Sigmoid())
        
    def forward(self, x, return_embedding=False):
        # x shape: (batch, seq_len, features)
        output, (hn, cn) = self.lstm(x)
        
        # Select final hidden state of the last layer
        feature = hn[-1, :, :]
        
        output = self.rul_regressor(feature)
        
        if return_embedding:
            return output.squeeze(1), feature

        return output.squeeze(1)


class Multilayer_LSTM_DANN(nn.Module):
    '''
    LSTM with Domain Adversarial Branch.
    '''
    def __init__(self, num_features=14, hidden_units=20, num_layers=3, n_neurons=50):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=num_layers
        )
        
        self.rul_regressor = nn.Sequential()
        self.rul_regressor.add_module('fc1', nn.Linear(in_features=hidden_units, out_features=n_neurons))
        self.rul_regressor.add_module('relu1', nn.ReLU(True))
        self.rul_regressor.add_module('fc2', nn.Linear(in_features=n_neurons, out_features=1))
        self.rul_regressor.add_module('sigmoid', nn.Sigmoid())
        
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('dc_fc1', nn.Linear(in_features=hidden_units, out_features=50))
        self.domain_classifier.add_module('dc_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('dc_fc2', nn.Linear(in_features=50, out_features=30))
        self.domain_classifier.add_module('dc_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('dc_fc3', nn.Linear(in_features=30, out_features=1))
        self.domain_classifier.add_module('dc_sigmoid', nn.Sigmoid())
        
    def forward(self, x, align_int=1, return_embedding=False):
        output, (hn, cn) = self.lstm(x)
        
        # Select final hidden state of the last layer
        feature = hn[-1, :, :]
        
        rul_output = self.rul_regressor(feature)

        if return_embedding:
            return rul_output.squeeze(1), feature
        
        # Gradient Reversal for DANN
        reverse_feature = ReverseLayerF.apply(feature, align_int)
        domain_output = self.domain_classifier(reverse_feature)
        
        return rul_output.squeeze(1), domain_output.squeeze(1)