#!/usr/bin/env python
# coding: utf-8
#Libraries Initialization

# In[147]:

import streamlit
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Here we could read CSV files that we have obtained from yahoo finance and these csv files contains 5 Year data about a companies
# opening,closing,high,low stock prices and the number of shares on that day

# In[148]:


ambuja = pd.read_csv('AMBUJACEM.NS.csv', index_col='Date', parse_dates=['Date'])
ambuja.head()


# In[149]:


itc = pd.read_csv('ITC.NS.csv', index_col='Date', parse_dates=['Date'])
itc.head()


# In[150]:


tcs = pd.read_csv('TCS.NS.csv', index_col='Date', parse_dates=['Date'])
tcs.head()


# In[151]:


dabur = pd.read_csv('DABUR.NS.csv', index_col='Date', parse_dates=['Date'])
ambuja.head()


# And we have stored these csv files into variables using pandas function called read_csv and we could perform various functions
# on those variables and we could plot graphs with those variables as well

# In[152]:


ambuja.describe()


# In[153]:


ambuja.info()


# In[154]:


ambuja.isna().sum()


# In[155]:


itc.describe()


# In[156]:


itc.columns


# In[157]:


tcs.isna().sum()


# Here we start to plot graphs about a companies opening,closing,high and low stock prices and volume for that day

# In[158]:


dabur['2018':'2023'].plot(subplots=True, figsize=(10,12))
plt.title('Dabur stock attributes from 2018 to 2023')
plt.show()
     


# In[159]:


itc['2018':'2023'].plot(subplots=True, figsize=(10,12))
plt.title('ITC Limited stock attributes from 2018 to 2023')
plt.show()


# In[160]:


ambuja['2018':'2023'].plot(subplots=True, figsize=(10,12))
plt.title('Ambuja Cement stock attributes from 2018 to 2023')
plt.show()


# In[161]:


tcs['2018':'2023'].plot(subplots=True, figsize=(10,12))
plt.title('TATA Consultancy Services stock attributes from 2018 to 2023')
plt.show()


# In the graph below we can see that TCS has been performing exceptionally well as compared to dabur,ambuja and ITC and that could because
# of the technology boom the market has been going through
# here we compare HIGH stock prices of all companies with each other 

# In[162]:


dabur.High.plot()
itc.High.plot()
ambuja.High.plot()
tcs.High.plot()
plt.legend(['Dabur','ITC Limited','Ambuja Cement','TATA Consultancy Services'])
plt.show()


# In[163]:


dabur_mean = dabur.High.expanding().mean()
dabur_std = dabur.High.expanding().std()
dabur.High.plot()
dabur_mean.plot()
dabur_std.plot()
plt.legend(['High','Expanding Mean','Expanding Standard Deviation'])
plt.title('Dabur')
plt.show()


# In[164]:


itc_mean = itc.High.expanding().mean()
itc_std = itc.High.expanding().std()
itc.High.plot()
itc_mean.plot()
itc_std.plot()
plt.legend(['High','Expanding Mean','Expanding Standard Deviation'])
plt.title('ITC Limited')
plt.show()


# In[165]:


ambuja_mean = ambuja.High.expanding().mean()
ambuja_std = ambuja.High.expanding().std()
ambuja.High.plot()
ambuja_mean.plot()
ambuja_std.plot()
plt.legend(['High','Expanding Mean','Expanding Standard Deviation'])
plt.title('Ambuja Cement')
plt.show()


# In[166]:


tcs_mean = tcs.High.expanding().mean()
tcs_std = tcs.High.expanding().std()
tcs.High.plot()
tcs_mean.plot()
tcs_std.plot()
plt.legend(['High','Expanding Mean','Expanding Standard Deviation'])
plt.title('TATA Consultancy Services')
plt.show()


# In[167]:


dabur_mean = dabur.Close.expanding().mean()
dabur_std = dabur.Close.expanding().std()
dabur.High.plot()
dabur_mean.plot()
dabur_std.plot()
plt.legend(['Close','Expanding Mean','Expanding Standard Deviation'])
plt.title('Dabur')
plt.show()


# In[168]:


itc_mean = itc.Close.expanding().mean()
itc_std = itc.Close.expanding().std()
itc.High.plot()
itc_mean.plot()
itc_std.plot()
plt.legend(['Close','Expanding Mean','Expanding Standard Deviation'])
plt.title('ITC Limited')
plt.show()


# In[169]:


ambuja_mean=ambuja.Close.expanding().mean()
ambuja_std=ambuja.Close.expanding().std()
ambuja.High.plot()
ambuja_mean.plot()
ambuja_std.plot()
plt.legend(['Close','Expanding Mean','Expanding Standard Deviation'])
plt.title('Ambuja Cements')
plt.show()


# In[170]:


tcs_mean = tcs.Close.expanding().mean()
tcs_std = tcs.Close.expanding().std()
tcs.High.plot()
tcs_mean.plot()
tcs_std.plot()
plt.legend(['Close','Expanding Mean','Expanding Standard Deviation'])
plt.title('TATA Consultancy Services')
plt.show()


# In[171]:


from pylab import rcParams
import statsmodels.api as sm
import matplotlib.pyplot as plt


# # Trend and Seasonality

# In[172]:


filepath = 'DABUR.NS.csv'
data_dabur = pd.read_csv(filepath)
data_dabur = data_dabur.sort_values('Date')
data_dabur.head() 


# In[173]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
plt.figure(figsize = (15,9))
plt.plot(data_dabur[['Close']])
plt.xticks(range(0,data_dabur.shape[0],500),data_dabur['Date'].loc[::500],rotation=45)
plt.title("dabur Stock Price",fontsize=18, fontweight='bold')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price (INR)',fontsize=18)
plt.show()
     


# In[174]:


filepath = 'ITC.NS.csv'
data_itc = pd.read_csv(filepath)
data_itc = data_itc.sort_values('Date')
data_itc.head()


# In[175]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
plt.figure(figsize = (15,9))
plt.plot(data_itc[['Close']])
plt.xticks(range(0,data_itc.shape[0],500),data_itc['Date'].loc[::500],rotation=45)
plt.title("ITC Limited Stock Price",fontsize=18, fontweight='bold')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price (INR)',fontsize=18)
plt.show()


# In[176]:


filepath = 'AMBUJACEM.NS.csv'
data_ambuja = pd.read_csv(filepath)
data_ambuja = data_ambuja.sort_values('Date')
data_ambuja.head()


# In[177]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
plt.figure(figsize = (15,9))
plt.plot(data_ambuja[['Close']])
plt.xticks(range(0,data_ambuja.shape[0],500),data_ambuja['Date'].loc[::500],rotation=45)
plt.title("AMBUJA CEMENT Stock Price",fontsize=18, fontweight='bold')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price (INR)',fontsize=18)
plt.show()


# In[178]:


filepath = 'TCS.NS.csv'
data_tcs = pd.read_csv(filepath)
data_tcs = data_tcs.sort_values('Date')
data_tcs.head()


# In[146]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
plt.figure(figsize = (15,9))
plt.plot(data_tcs[['Close']])
plt.xticks(range(0,data_tcs.shape[0],500),data_tcs['Date'].loc[::500],rotation=45)
plt.title("TATA Consulatancy Services Stock Price",fontsize=18, fontweight='bold')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price (INR)',fontsize=18)
plt.show()


# In[179]:


price_dabur = data_dabur[['Close']]
price_dabur.info()


# In[181]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
price_dabur['Close'] = scaler.fit_transform(price_dabur['Close'].values.reshape(-1,1))


# In[182]:


def split_data(stock, lookback):
    data_raw = stock.to_numpy()
    data = []
    
    
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]
     


# In[184]:


lookback = 20 
x_train, y_train, x_test, y_test = split_data(price_dabur, lookback)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)
     


# In[185]:


import torch
import torch.nn as nn

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)
     


# In[186]:


input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 105


# In[187]:


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out


# In[188]:


model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)


# In[189]:


import time 
hist = np.zeros(num_epochs)
start_time = time.time()
gru = []

for t in range(num_epochs):
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train_gru)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

training_time = time.time()-start_time    
print("Training time: {}".format(training_time))
     


# In[190]:


predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
original = pd.DataFrame(scaler.inverse_transform(y_train_gru.detach().numpy()))


# In[191]:


import seaborn as sns
sns.set_style("darkgrid")    

fig = plt.figure()
fig.subplots_adjust(hspace=0.2, wspace=0.2)

plt.subplot(1, 2, 1)
ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (GRU)", color='tomato')
ax.set_title('Dabur stock price', size = 14, fontweight='bold')
ax.set_xlabel("Days", size = 14)
ax.set_ylabel("Cost (INR)", size = 14)
ax.set_xticklabels('', size=10)


plt.subplot(1, 2, 2)
ax = sns.lineplot(data=hist, color='royalblue')
ax.set_xlabel("Epoch", size = 14)
ax.set_ylabel("Loss", size = 14)
ax.set_title("Training Loss", size = 14, fontweight='bold')
fig.set_figheight(6)
fig.set_figwidth(16)
     


# In[192]:


import math, time
from sklearn.metrics import mean_squared_error


y_test_pred = model(x_test)


y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train_gru.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test_gru.detach().numpy())


trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
gru.append(trainScore)
gru.append(testScore)
gru.append(training_time)


# In[194]:


trainPredictPlot = np.empty_like(price_dabur)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred


testPredictPlot = np.empty_like(price_dabur)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(y_train_pred)+lookback-1:len(price_dabur)-1, :] = y_test_pred

original = scaler.inverse_transform(price_dabur['Close'].values.reshape(-1,1))

predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
predictions = np.append(predictions, original, axis=1)
result = pd.DataFrame(predictions)
     


# In[196]:


import plotly.express as px
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
                    mode='lines',
                    name='Train prediction')))
fig.add_trace(go.Scatter(x=result.index, y=result[1],
                    mode='lines',
                    name='Test prediction'))
fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
                    mode='lines',
                    name='Actual Value')))
fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=False,
        linecolor='white',
        linewidth=2
    ),
    yaxis=dict(
        title_text='Close (USD)',
        titlefont=dict(
            family='Rockwell',
            size=12,
            color='white',
            ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
    ),
    showlegend=True,
    template = 'plotly_dark'

)



annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Dabur Stock Prediction',
                              font=dict(family='Rockwell',
                                        size=26,
                                        color='white'),
                              showarrow=False))
fig.update_layout(annotations=annotations)

fig.show()


# In[198]:


price_itc = data_itc[['Close']]
price_itc.info()


# In[199]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
price_itc['Close'] = scaler.fit_transform(price_itc['Close'].values.reshape(-1,1))


# In[201]:


lookback = 20
x_train, y_train, x_test, y_test = split_data(price_itc, lookback)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)


# In[202]:


import torch
import torch.nn as nn

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)
     


# In[203]:


model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)


# In[204]:


hist = np.zeros(num_epochs)
start_time = time.time()
gru = []

for t in range(num_epochs):
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train_gru)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

training_time = time.time()-start_time    
print("Training time: {}".format(training_time))


# In[205]:


predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
original = pd.DataFrame(scaler.inverse_transform(y_train_gru.detach().numpy()))


# In[206]:


import seaborn as sns
sns.set_style("darkgrid")    

fig = plt.figure()
fig.subplots_adjust(hspace=0.2, wspace=0.2)

plt.subplot(1, 2, 1)
ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (GRU)", color='tomato')
ax.set_title('ITC Limited stock price', size = 14, fontweight='bold')
ax.set_xlabel("Days", size = 14)
ax.set_ylabel("Cost (INR)", size = 14)
ax.set_xticklabels('', size=10)


plt.subplot(1, 2, 2)
ax = sns.lineplot(data=hist, color='royalblue')
ax.set_xlabel("Epoch", size = 14)
ax.set_ylabel("Loss", size = 14)
ax.set_title("Training Loss", size = 14, fontweight='bold')
fig.set_figheight(6)
fig.set_figwidth(16)


# In[207]:


import math, time
from sklearn.metrics import mean_squared_error

y_test_pred = model(x_test)


y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train_gru.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test_gru.detach().numpy())

trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
gru.append(trainScore)
gru.append(testScore)
gru.append(training_time)


# In[208]:


trainPredictPlot = np.empty_like(price_itc)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred

# shift test predictions for plotting
testPredictPlot = np.empty_like(price_itc)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(y_train_pred)+lookback-1:len(price_itc)-1, :] = y_test_pred

original = scaler.inverse_transform(price_itc['Close'].values.reshape(-1,1))

predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
predictions = np.append(predictions, original, axis=1)
result = pd.DataFrame(predictions)
     


# In[210]:


import plotly.express as px
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
                    mode='lines',
                    name='Train prediction')))
fig.add_trace(go.Scatter(x=result.index, y=result[1],
                    mode='lines',
                    name='Test prediction'))
fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
                    mode='lines',
                    name='Actual Value')))
fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=False,
        linecolor='white',
        linewidth=2
    ),
    yaxis=dict(
        title_text='Close (INR)',
        titlefont=dict(
            family='Rockwell',
            size=12,
            color='white',
            ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
    ),
    showlegend=True,
    template = 'plotly_dark'

)



annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='ITC Limited Stock Prediction',
                              font=dict(family='Rockwell',
                                        size=26,
                                        color='white'),
                              showarrow=False))
fig.update_layout(annotations=annotations)

fig.show()
     


# In[211]:


price_ambuja = data_ambuja[['Close']]
price_ambuja.info()


# In[213]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
price_ambuja['Close'] = scaler.fit_transform(price_ambuja['Close'].values.reshape(-1,1))


# In[215]:


lookback = 20 # choose sequence length
x_train, y_train, x_test, y_test = split_data(price_ambuja, lookback)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)


# In[216]:


import torch
import torch.nn as nn

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)


# In[217]:


input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 105


# In[218]:


model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
     


# In[219]:


hist = np.zeros(num_epochs)
start_time = time.time()
gru = []

for t in range(num_epochs):
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train_gru)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

training_time = time.time()-start_time    
print("Training time: {}".format(training_time))


# In[220]:


predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
original = pd.DataFrame(scaler.inverse_transform(y_train_gru.detach().numpy()))
     


# In[221]:


import seaborn as sns
sns.set_style("darkgrid")    

fig = plt.figure()
fig.subplots_adjust(hspace=0.2, wspace=0.2)

plt.subplot(1, 2, 1)
ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (GRU)", color='tomato')
ax.set_title('Ambuja Cement stock price', size = 14, fontweight='bold')
ax.set_xlabel("Days", size = 14)
ax.set_ylabel("Cost (INR)", size = 14)
ax.set_xticklabels('', size=10)


plt.subplot(1, 2, 2)
ax = sns.lineplot(data=hist, color='royalblue')
ax.set_xlabel("Epoch", size = 14)
ax.set_ylabel("Loss", size = 14)
ax.set_title("Training Loss", size = 14, fontweight='bold')
fig.set_figheight(6)
fig.set_figwidth(16)


# In[222]:


import math, time
from sklearn.metrics import mean_squared_error

# make predictions
y_test_pred = model(x_test)

# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train_gru.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test_gru.detach().numpy())

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
gru.append(trainScore)
gru.append(testScore)
gru.append(training_time)


# In[225]:


trainPredictPlot = np.empty_like(price_ambuja)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred


testPredictPlot = np.empty_like(price_ambuja)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(y_train_pred)+lookback-1:len(price_ambuja)-1, :] = y_test_pred

original = scaler.inverse_transform(price_ambuja['Close'].values.reshape(-1,1))

predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
predictions = np.append(predictions, original, axis=1)
result = pd.DataFrame(predictions)


# In[226]:


import plotly.express as px
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
                    mode='lines',
                    name='Train prediction')))
fig.add_trace(go.Scatter(x=result.index, y=result[1],
                    mode='lines',
                    name='Test prediction'))
fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
                    mode='lines',
                    name='Actual Value')))
fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=False,
        linecolor='white',
        linewidth=2
    ),
    yaxis=dict(
        title_text='Close (INR)',
        titlefont=dict(
            family='Rockwell',
            size=12,
            color='white',
            ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
    ),
    showlegend=True,
    template = 'plotly_dark'

)



annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Ambuja Cement Stock Prediction',
                              font=dict(family='Rockwell',
                                        size=26,
                                        color='white'),
                              showarrow=False))
fig.update_layout(annotations=annotations)

fig.show()
     


# In[227]:


price_tcs = data_tcs[['Close']]
price_tcs.info()


# In[228]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
price_tcs['Close'] = scaler.fit_transform(price_tcs['Close'].values.reshape(-1,1))


# In[230]:


lookback = 20
x_train, y_train, x_test, y_test = split_data(price_tcs, lookback)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)


# In[237]:


input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 105


# In[238]:


model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
     


# In[239]:


hist = np.zeros(num_epochs)
start_time = time.time()
gru = []

for t in range(num_epochs):
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train_gru)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

training_time = time.time()-start_time    
print("Training time: {}".format(training_time))
     


# In[240]:


predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
original = pd.DataFrame(scaler.inverse_transform(y_train_gru.detach().numpy()))


# In[242]:


import seaborn as sns
sns.set_style("darkgrid")    

fig = plt.figure()
fig.subplots_adjust(hspace=0.2, wspace=0.2)

plt.subplot(1, 2, 1)
ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (GRU)", color='tomato')
ax.set_title('TATA Consultancy Services stock price', size = 14, fontweight='bold')
ax.set_xlabel("Days", size = 14)
ax.set_ylabel("Cost (INR)", size = 14)
ax.set_xticklabels('', size=10)


plt.subplot(1, 2, 2)
ax = sns.lineplot(data=hist, color='royalblue')
ax.set_xlabel("Epoch", size = 14)
ax.set_ylabel("Loss", size = 14)
ax.set_title("Training Loss", size = 14, fontweight='bold')
fig.set_figheight(6)
fig.set_figwidth(16)
     


# In[243]:


import math, time
from sklearn.metrics import mean_squared_error

y_test_pred = model(x_test)


y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train_gru.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test_gru.detach().numpy())


trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
gru.append(trainScore)
gru.append(testScore)
gru.append(training_time)


# In[244]:


trainPredictPlot = np.empty_like(price_tcs)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred


testPredictPlot = np.empty_like(price_tcs)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(y_train_pred)+lookback-1:len(price_tcs)-1, :] = y_test_pred

original = scaler.inverse_transform(price_tcs['Close'].values.reshape(-1,1))

predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
predictions = np.append(predictions, original, axis=1)
result = pd.DataFrame(predictions)
     


# In[245]:


import plotly.express as px
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
                    mode='lines',
                    name='Train prediction')))
fig.add_trace(go.Scatter(x=result.index, y=result[1],
                    mode='lines',
                    name='Test prediction'))
fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
                    mode='lines',
                    name='Actual Value')))
fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=False,
        linecolor='white',
        linewidth=2
    ),
    yaxis=dict(
        title_text='Close (INR)',
        titlefont=dict(
            family='Rockwell',
            size=12,
            color='white',
             ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
    ),
    showlegend=True,
    template = 'plotly_dark'

)



annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='TATA Consultancy Services Stock Prediction',
                              font=dict(family='Rockwell',
                                        size=26,
                                        color='white'),
                              showarrow=False))
fig.update_layout(annotations=annotations)

fig.show()


# In[ ]:




