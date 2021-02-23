<h2>Motivation</h2>
Use SVM for machine learning using a library - sklearn

<h2>Content</h2>
<table>
  <tr>
    <th>File</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>linear_decision_boundary.py</td>
    <td>
    <ul>
    <li>use linear kernel for classification</li>
    <li>show the effect of regularization via C. large C overfit - score is 100% linear_decision_boundary_C_100.png. small C is more robust - margin but score is less than 100% linear_decision_boundary_C_1.png</li>
    <li>the hypothesis is h=sigmoid(z) where z is teta0+teta1*x1+teta2*x2 . from sigmoid we know that the decision is done on z=0 thus we plot teta0+teta1*x1+teta2*x2=0</li>
  </tr>
  <tr>
    <td>non_linear_decision_boundary.py</td>
    <td>
    <ul>
    <li>use nonlinear kernel - guasian for classification</li>
    <li>show the effect of regularization via C. large C overfit  non_linear_decision_boundary_C_100.png. small C is more robust  non_linear_decision_boundary_C_1.png</li>
    <li>the decision boundary is computed using contour</li>
  </tr>
</table>
