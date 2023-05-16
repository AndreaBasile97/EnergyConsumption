# Energy Consumption Project 🔌

## 🥅 Goal 
The goal of this project is to predict the energy consumptio for each house.
The differeces lies on the type of predictions. In order to do these kind of operations we need two types of infos:

- Energy consumed for a specific slice of time.
- Spatial information of the houses.

The dataset is built in order to extract these data.

---

### ⏲️ Temporal Autocorrelation 

The energy consumption could change in months (example: In January the energy consumption could be different from August). So keep tracking of these consumptions is informative. 

Below are shown two types of settings that change the structure on how the ML model will receive the informations and what kind of information should predict:

- **Single-Step**: Given descriptive features of *n* months, predict the energy for the *n+1* month. Recursively, we can add the predicted consumption to the descriptive features and predict the *n+2* month and so on.

- **Multi-Step** ⭐: MIMO approach: Multi-input, Multi-output. With this method we try to predict multiple energy consumptions for multiple months in one phase. The MS settings gave better results in term of scores and for this reason, the goal can be extended in term of granularity of the time, passing from predicting energy consumption for months to predict energy for specific slice of day time.

⭐: *best results on the project*

---

### 🗺️ Spatial Autocorrelation

Another important aspect that could be helpful to predict the energy is the location in the world. We expect, for example, that people who live in north regions of the world will have different behaviors in terms of energy consumption compared to people who lives in warmer places. 

The problem now is, how to let our model to understand the spatial informations between the houses. This is achieved thanks to two techniques:

- **LISA** : is an indicator value that highlight the correlation in terms of consumption based on the distance between consumers. Basically, this indicator will tell us how much, the energy consumption of a customer *A* is similar to the energy consumption of *B* knowing that *A* and *B* are neighborgs or not. This is done through tha Neighborood matrix which will tell what are the customers that are close each other. then, we normalize the matrix and each descriptive features of the consumers. Then, the indicator score $I$ is computed for each single consumer and added as additional *feature descriptors*.

$$ N[c_a, c_b] = \begin{cases} 
    1 & \text{if } dist(c_a, c_b)<maxDist\\
    0 & \text{otherwise } \\
\end{cases} 
$$

$$ ... \text{ normalize N } and \text{ x }... $$

$$ I_{x,c_a} = x'_{c_a} \sum (N'[c_a, c_i]* x'_{c_i}) $$


Note: $maxDist$ is an hyperparameter.

- **PCNM** ⭐: This approach is slightly different from LISA. Basically we need $D$ that is the matrix containing the Harvesine-Distance between consumers. Then compute a truncated distance matrix $D^*$ and perform the **PCoA** (Principal Coordinate Analysis) on $D^*$. Computing the PCoA


$$ D^* = \begin{cases} 
    dist(c_a, c_b) & \text{if } dist(c_a, c_b) <= maxDist\\
    4 * maxDist & \text{otherwise } \\
\end{cases} 
$$

On the matrix $D^*$ compute the PCoA that will output a matrix containing on the rows, all the consumers and on the columns the PCoA eigen-vectors. We are interested only in eigen-vectors associated with high-positive eigen-values since they represent an high positive autocorrelation. The eigenvectors with these characteristics will be kept and used as additional *feature descriptors*.

⭐: *best results on the project*

---

<br>
<br>

# 📂 The Dataset 

The dataset used in this project is different from the original since this is based on more granular way in terms of times, infact, now the energy consumption is taken every 15 minutes for each consumer.


<table>
    <thead>
        <tr>
            <th>house</th>
            <th>date</th>
            <th>grid</th>
            <th>solar</th>
            <th>city</th>
            <th>house_con...</th>
            <th>total_squ...</th>
            <th>lat</th>
            <th>lon</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1222</td>
            <td>2019-05-01 00:00:00-05:00</td>
            <td>0.159</td>
            <td>-0.005</td>
            <td>Ithaca</td>
            <td>1958</td>
            <td>1750</td>
            <td>42.444</td>
            <td>-76.500</td>
        </tr>
        <tr>
            <td>1222</td>
            <td>2019-05-01 00:15:00-05:00</td>
            <td>2.879</td>
            <td>-0.004</td>
            <td>Ithaca</td>
            <td>1958</td>
            <td>1750</td>
            <td>42.444</td>
            <td>-76.500</td>
        </tr>
        <tr>
            <td>...</td>
            <td>...</td>
            <td>...</td>
            <td>...</td>
            <td>...</td>
            <td>...</td>
            <td>...</td>
            <td>...</td>
            <td>...</td>
        </tr>
        <tr>
            <td>1222</td>
            <td>2019-10-31 23:45:00-05:00</td>
            <td>0.162</td>
            <td>-0.005</td>
            <td>Ithaca</td>
            <td>1958</td>
            <td>1750</td>
            <td>42.444</td>
            <td>-76.500</td>
        </tr>
    </tbody>
</table>

We have 96 slices of times/day for each costumer. The data are extract for an interval of 184 days so we have a total of 17.664 values of energy consumption for each consumer.

<br>
<br>

# 🔧 Configurator

The configurator is a tool to transform the dataset extracted by the .csv file in order to set it for the training phase. The Configurator is a class containing these attributes:

- *configuration*: Single-Step, Multi-Step.
- *windows_size*: How much 'time-steps' back we want to keep.
- *n_targets*: used in Multi-step to specify how many future time steps we want predict.
- *target*: the name of the variable considered as target to predict.
- *key*: the key that distinguish a customer from another one.
- *histFeatures*: what historical features add to dataset. The features specified here are added using 'lag' (past values) based on window-size.
- *dateCol*: specify the name of the date column.
- *spatial_method*: LISA, PCNM

Then there are 7 methods:

- *spatial*: apply spatial transformation in order to compute new features like the indicator if using LISA or eigenvectors associated to high postive eigen-values if using PCNM.
- *add_feature*: method useful to apply historical features in the dataset specifying also the type of features 'lag' (past) or 'lead' (future).
- *Local_Moran*: compute the Local Moran statistics that measures how much, a value is spatially correlated to neighbors.
- *transform*: manipulate the dataset in order to make it fisible to our task.
- *prediction*: generate prediction for target features based on the setting (SS, MT) if MT then, the MT_learning_prediction is used.
- *MT_learning_prediction*: Multi-target learning prediction. Used when the model is required to predict multiple outputs variables. This method split the dataset in train, validation and test. Learn from the data and generate prediction.
- *self_learning_prediction*: The same of MT learning prediction but for SS setting. Infact it will learn from the past data and will genereate one single prediction which will be added to past data and will be used for make future predictions in a recursive way.