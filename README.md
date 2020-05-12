# Gender classification based on finger print images

## Intro

## Methods
### Pre-processing
insert pictures regular images
insert pictures enhanced image
 
### Data augmentation
insert examples of augmented images

### Tuning
explain hyperopt process and what parameters are optimized for
 
### Evaluation
accuracy / confusion matrix (explain precision and recall)
 
 
## Results 
<table>
<thead>
  <tr>
    <th rowspan="2"><br>Model</th>
    <th colspan="4">Data augmentation</th>
    <th colspan="4">No data augmentation</th>
  </tr>
  <tr>
    <td colspan="2">Regular dataset</td>
    <td colspan="2">Enhanced dataset</td>
    <td colspan="2">Regular dataset</td>
    <td colspan="2">Enhanced dataset</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td>Validation</td>
    <td>Test</td>
    <td>Validation</td>
    <td>Test</td>
    <td>Validation</td>
    <td>Test</td>
    <td>Validation</td>
    <td>Test</td>
  </tr>
  <tr>
    <td>VGG-16</td>
    <td>69.50%</td>
    <td>68.91%</td>
    <td>69.67%</td>
    <td>66.91%</td>
    <td>70.00%</td>
    <td>64.78%</td>
    <td>69.17%</td>
    <td>67.39%</td>
  </tr>
  <tr>
    <td>ResNet-18</td>
    <td>65.50%</td>
    <td>61.74%</td>
    <td>68.33%</td>
    <td>64.35%</td>
    <td>66.00%</td>
    <td>64.78%</td>
    <td>65.17%</td>
    <td>65.87%</td>
  </tr>
  <tr>
    <td>ResNet-34</td>
    <td>68.17%</td>
    <td>66.09%</td>
    <td>68.33%</td>
    <td>68.33%</td>
    <td>72.00%</td>
    <td>64.78%</td>
    <td>67.67%</td>
    <td>68.48%</td>
  </tr>
</tbody>
</table>

insert confusion matrix of best performing model
