# main.py

# Necessary Condition Analysis (NCA) allows researchers and practitioners to
# identify necessary (but not sufficient) conditions in data sets.

# Reference:
# Dul, J. (2016) "Necessary Condition Analysis (NCA):
# Logic and Methodology of 'Necessary but Not Sufficient' Causality."
# Organizational Research Methods 19(1), 10-52.
# https://journals.sagepub.com/doi/pdf/10.1177/1094428115584005

# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Definition of main NCA class
class NCA:
  """
  A class to perform Necessary Condition Analysis on a given dataset

  ...

  Attributes
  ----------
  ceilings : list
    The selected ceiling lines to fit the data to the NCA

  Methods
  -------
  

  """
  # Initialize only requires the ceilings to be used in NCA analysis
  def __init__(self, ceilings=['ce-fdh', 'cr-fdh', 'ols']):
    """
    Parameters
    ----------
    ceilings : list
      A list of ceiling lines from which to calculate NCA parameters 
      (default is: ['ce-fdh', 'cr-fdh', 'ols'])
    """
    self.ceilings = ceilings
    self.__fitted = False
    self.__n_dec = 2

  # Helper functions for the fit() method
  ## Validate correct shape of data (TODO)
  @staticmethod
  def __check_data(X, y, data):
    pass#print("Checking data...")

  ## Calculate initial parameters for each determinant and the output variable
  @staticmethod
  def __define_scope_lims(data, X, y):
    scope_lims = {}
    for x_elem in X:
      scope_lims[x_elem]={
          "n_obs":len(data),
          "Xmin": data[x_elem].min(),
          "Xmax": data[x_elem].max(),
          "Ymin": data[y].min(),
          "Ymax": data[y].max()
          }
      scope_lims[x_elem]["Scope"] = (scope_lims[x_elem]["Xmax"]-scope_lims[x_elem]["Xmin"])*(scope_lims[x_elem]["Ymax"]-scope_lims[x_elem]["Ymin"])
    return scope_lims

  ## Calculate sorted arrays: Pre-processing of data before using it in NCA functions
  @staticmethod
  def __create_sorted_arrays(data, X, y):
    sorted_arrays = {}
    for x_elem in X:
      ## Create a dataframe with the determinant and the outcome and sort them from smallest to largest in X and Y
      sorted_df = data[[x_elem,y]].sort_values(by=[x_elem,y])
      #sorted_index = sorted_df.index
      ## Change it to numpy array for better handling
      sorted_array = sorted_df.to_numpy()
      ## Add it to sorted_arrays dict
      sorted_arrays[x_elem] = sorted_array
    return sorted_arrays

  ## Calculate CE-FDH envelope list
  @staticmethod
  def __CE_FDH_envelope_list(sorted_array):
    ## Define index to keep track of iteration
    idx_num = 1
    ## Define first edge value and create envelope list
    envelope_list = np.array([sorted_array[0]])
    ## Iterate over each entry in the sorted array to find the envelope list
    ## elements (values that create the ceiling piecewise function in CE-FDH)
    for pair in sorted_array:
      pair_x_val = pair[0]
      pair_y_val = pair[1]
      if idx_num == 1:
        ## Define initial conditions to evaluate envelope list
        current_x_value = pair_x_val
        current_y_value = pair_y_val
      else:
        ## Move vertically upward to the observation with the largest Y for same X
        if pair_x_val == current_x_value:
          if pair_y_val > current_y_value:
            current_y_value = pair_y_val
            envelope_list = np.append(envelope_list,[[pair_x_val, pair_y_val]], axis=0)
        ## Move horizontally to the right until a point with a larger Y
        elif pair_y_val >= current_y_value:
          ## Next line: stepwise function increases
          envelope_list = np.append(envelope_list,[[pair_x_val, current_y_value]], axis=0)
          current_y_value = pair_y_val
          current_x_value = pair_x_val
          ## Next line: stepwise function breakpoint
          envelope_list = np.append(envelope_list,[[pair_x_val, pair_y_val]], axis=0)
        ## Move horizontally to the last point (x_max) if it has a lower y than the current x
        elif idx_num == len(sorted_array):
          envelope_list = np.append(envelope_list,[[pair_x_val, current_y_value]], axis=0)
        ## Discard observations below current piecewise line
        else:
          pass
      ## Increase index
      idx_num += 1

    return envelope_list

  # Compute CE-FDH Upper-left edges(north-west corners)
  @staticmethod
  def __CE_FDH_peers(envelope_list):
    ## Create an empty array to store upper left edges
    upper_left_edges = np.empty((0,2))
    ## Iterate over the envelope list to find the upper left edges, that is
    ## the points where a vertical part of the CE-FDH (piecewise linear function)
    ## ends and continues as a horizontal line when X increases
    for idx in range(1,envelope_list.shape[0]):
      ## If it continues as a horizontal line, append previous point to the array of upper left edges
      if envelope_list[idx][0] > envelope_list[idx-1][0]:
        upper_left_edges = np.append(upper_left_edges,[envelope_list[idx-1]],axis=0)
      ## If it reaches to the last point without fulfilling the previous condition, append it
      elif (idx == (envelope_list.shape[0]-1)) & (envelope_list[idx][1] > envelope_list[idx-1][1]):
        upper_left_edges = np.append(upper_left_edges,[envelope_list[idx]],axis=0)
    return upper_left_edges

  # Compute CE-FDH effect size and the size of the ceiling zone
  @staticmethod
  def __CE_FDH_effect_size(scope_lims,upper_left_edges):
    # Get scope values for x and y
    Xmin = scope_lims["Xmin"]
    Xmax = scope_lims["Xmax"]
    Ymin = scope_lims["Ymin"]
    Ymax = scope_lims["Ymax"]
    Scope = scope_lims["Scope"]
    ## Create a new array to compute the area below the ceiling line
    upper_corners = upper_left_edges
    ## Append aditional point if Xmax > last edge corner on upper_left_edges
    if upper_corners[(len(upper_corners)-1),0] < Xmax:
      fict_corner = np.array([Xmax,Ymax]) # Fictious corner to include on the upper corners array
      upper_corners = np.append(upper_corners, [fict_corner], axis=0)
    ## Compute area below ceiling line by building rectangles from the upper left edges
    area_blw_CL = 0
    for idx in range(1,upper_corners.shape[0]):
      delta_x = upper_corners[idx][0]-upper_corners[idx-1][0]
      delta_y = upper_corners[idx-1][1]-Ymin ## !!! FIX Possible rounding errors !!!
      partial_area_blw_CL = delta_x*delta_y
      area_blw_CL += partial_area_blw_CL
    ## Compute Effect Size of NC
    d_CE_FDH = ((Scope-area_blw_CL) / Scope).round(3)
    return d_CE_FDH, Scope-area_blw_CL

  # Compute FR-FDH OLS model parameters (a + b*x) returns (slope, intercept)
  @staticmethod
  def __CR_FDH_OLS_params(CE_FDH_upper_left_edges):
    return np.polyfit(CE_FDH_upper_left_edges[:,0], CE_FDH_upper_left_edges[:,1], deg=1)

  # Calculate CR-FDH polygon within ceiling line and scope area
  @staticmethod
  def __CR_FDH_polygon_array(scope_lims, b_CR_FDH, a_CR_FDH):
    # Get scope values for x and y
    Xmin = scope_lims["Xmin"]
    Xmax = scope_lims["Xmax"]
    Ymin = scope_lims["Ymin"]
    Ymax = scope_lims["Ymax"]
    Scope = scope_lims["Scope"]
    # Create an array with the initial point of the scope
    CR_FDH_Polygon = np.array([[Xmax,Ymin]])
    ## Placeholders for slope and intercept in OLS regression line
    m = b_CR_FDH
    b = a_CR_FDH
    ## TODO: Validate if regression line is within scope bounds
    ## Calculate points Y1 and Y2 on Xmin and Xmax for OLS Model (CR-FDH)
    y1 = m*Xmin + b
    y2 = m*Xmax + b
    ## Create polygon based on geometric features (intercept of OLS line and scope area)
    ### (Create polygon counter-clockwise) Validate location of y2
    if(y2 > Ymax):
      CR_FDH_Polygon = np.append(CR_FDH_Polygon, [[Xmax, Ymax]], axis=0)
      CR_FDH_Polygon = np.append(CR_FDH_Polygon, [[(Ymax-b)/m, Ymax]], axis=0)
    else:
      CR_FDH_Polygon = np.append(CR_FDH_Polygon, [[Xmax, y2]], axis=0)
    ### Validate location of y1
    if(y1 > Ymin):
      CR_FDH_Polygon = np.append(CR_FDH_Polygon, [[Xmin, y1]], axis=0)
      CR_FDH_Polygon = np.append(CR_FDH_Polygon, [[Xmin, Ymin]], axis=0)
    else:
      CR_FDH_Polygon = np.append(CR_FDH_Polygon, [[(Ymin-b)/m, Ymin]], axis=0)

    return CR_FDH_Polygon

  # Define CR-FDH helper function to calculate area in a polygon
  @staticmethod
  def __PolyArea(x,y):
      ## Based on the shoelace formula (https://stackoverflow.com/a/30408825/15548668)
      return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

  # Calculate CR-FDH effect size and the size of the ceiling zone
  @staticmethod
  def __CR_FDH_effect_size(Scope, CR_FDH_Polygon):
    ## Calculate area below CR line
    area_blw_CR =  NCA.__PolyArea(CR_FDH_Polygon[:,0],CR_FDH_Polygon[:,1])
    ## Calculate effect size
    d_CR_FDH = ((Scope-area_blw_CR)/Scope).round(3)
    return d_CR_FDH, Scope-area_blw_CR

  # Calculate accuracy from abline
  # Helper CR-FDH function
  @staticmethod
  def __accuracy_from_abline(data_array, slope_abline, intercept_abline):
    ## Create an empty array to store the values above the ceiling line
    above_abline = np.empty((0,2))
    ## Iterate over each point in the dataset
    for idx in range(0,data_array.shape[0]):
      # Calculate if current y point is above abline and append to array
      if data_array[idx][1] > (intercept_abline + slope_abline*data_array[idx][0]):
        above_abline = np.append(above_abline,[data_array[idx]],axis=0)
    ## Define a variable to store the accuracy
    accuracy_val = 1-(above_abline.shape[0] / data_array.shape[0])
    return(accuracy_val)

  # Calculate condition inefficiency from abline
  # Helper CR-FDH function
  @staticmethod
  def __condition_inefficiency_from_abline(scope_lims, slope_abline, intercept_abline):
    # Get scope values for x and y
    Xmin = scope_lims["Xmin"]
    Xmax = scope_lims["Xmax"]
    Ymin = scope_lims["Ymin"]
    Ymax = scope_lims["Ymax"]
    ## Placeholders for slope and intercept in OLS regression line
    m = slope_abline # Slope
    b = intercept_abline # Intercept
    ## Calculate points Y1 and Y2 on Xmin and Xmax for OLS Model
    y_for_Xmax = m*Xmax + b
    y_for_Xmin = m*Xmin + b
    # Calculate the condition inneficiency point in the x axis
    if y_for_Xmax > Ymax:
      X_Cmax = (Ymax-b)/m
      cond_ineff_point = [X_Cmax,Ymax]
    else:
      X_Cmax = Xmax
      cond_ineff_point = [X_Cmax,y_for_Xmax]
    ## Calculate condition inneficiency value
    i_subx = (Xmax-X_Cmax)/(Xmax-Xmin)
    return i_subx, cond_ineff_point

  # Calculate outcome inefficiency from abline
  # Helper CR-FDH function
  @staticmethod
  def __outcome_inefficiency_from_abline(scope_lims, slope_abline, intercept_abline):
    # Get scope values for x and y
    Xmin = scope_lims["Xmin"]
    Xmax = scope_lims["Xmax"]
    Ymin = scope_lims["Ymin"]
    Ymax = scope_lims["Ymax"]
    ## Placeholders for slope and intercept in OLS regression line
    m = slope_abline # Slope
    b = intercept_abline # Intercept
    ## Calculate points Y1 and Y2 on Xmin and Xmax for OLS Model
    y_for_Xmax = m*Xmax + b
    y_for_Xmin = m*Xmin + b
    # Outcome inefficiency
    ## Find Y_Cmin, the min value at which Y is not constrained by X
    if y_for_Xmin < Ymin:
      Y_Cmin = Ymin
      outc_ineff_point = [(Ymin-b)/m, Y_Cmin]
    else:
      Y_Cmin = y_for_Xmin
      outc_ineff_point = [Xmin, Y_Cmin]
    ## Calculate outcome inneficiency value
    i_suby = (Y_Cmin - Ymin) / (Ymax-Ymin)
    return i_suby, outc_ineff_point

  # Define helper function to retreive actual values from percentiles
  # (assuming a linear relationship) and percentiles from actual values
  # Helper functions in bottleneck tables
  @staticmethod
  def __actual_value(perc,max_val, min_val):
    return((max_val-min_val)*(perc/100)+min_val)

  @staticmethod
  def __perc_value(act,max_val, min_val,n_dec):
    if act == "NN":
      return("NN")
    else:
      return(np.round(100*(act-min_val)/(max_val-min_val),n_dec))

  @staticmethod
  def __bottleneck_x_calculation(actual_y, intercept, slope, xminval,n_dec):
    # Calculates current X value from Y value
    actual_x = np.round((actual_y-intercept)/slope,n_dec)
    if actual_x < xminval:
      actual_x = "NN"
    return(actual_x)

  # Bottleneck table for all variables in X for a given ceiling abline
  @staticmethod
  def __bottleneck_table_abline_ceiling(X,y,bottleneck_type, OLS_ceiling_line_dict, scope_lims_dict,n_dec):
    # This is a tabluar representation of the ceiling lines of one or more NC
    # It shows the required necessary level of the conditioins (as %)
    # for a given level of the outcome (as %)
    ## Define a range of percentiles to create bottleneck table
    outcome_perc = range(0,101,10)
    bb_table = pd.DataFrame({y+"_perc": outcome_perc})
    ## Define a range of percentiles to create bottleneck table
    ## Calculate minumum and maximum values for y
    Ymin = scope_lims_dict[list(scope_lims_dict.keys())[0]]["Ymin"]
    Ymax = scope_lims_dict[list(scope_lims_dict.keys())[0]]["Ymax"]
    ## From the percentiles, compute actual values
    bb_table[y+"_vals"] = bb_table[y+"_perc"].apply(NCA.__actual_value, args=(Ymax, Ymin))
    ## Calculate bottleneck values for determinants
    for x_item in X:
      bb_table[x_item+"_vals"] = bb_table[y+"_vals"].apply(NCA.__bottleneck_x_calculation,
                                                      args=(OLS_ceiling_line_dict[x_item][1],
                                                            OLS_ceiling_line_dict[x_item][0],
                                                            scope_lims_dict[x_item]["Xmin"],
                                                            n_dec))
    ## Calculate percentiles from actual values (determinants)
    for x_item in X:
      bb_table[x_item+"_perc"] = bb_table[x_item+"_vals"].apply(NCA.__perc_value,
                                                              args=(scope_lims_dict[x_item]["Xmax"],
                                                                    scope_lims_dict[x_item]["Xmin"],
                                                                    n_dec))
    # Drop columns according to condition
    if bottleneck_type == "actual":
        ## Drop percentage value columns
      bb_table = bb_table.filter(like="_vals", axis=1)
      bb_table = bb_table.rename(columns = lambda x: x.strip('_vals'))
    else:
      ## Drop actual value columns
      bb_table = bb_table.filter(like="_perc", axis=1)
      bb_table = bb_table.rename(columns = lambda x: x.strip('_perc'))
    return(bb_table)

  # Calculate OLS model for all observations (a + b * x) returns (slope, intercept)
  @staticmethod
  def __OLS_params(data_array):
    return np.polyfit(data_array[:,0], data_array[:,1], deg=1)

  # Define properties for non-public attributes  
  ## scope_ property
  @property
  def scope_(self):
    """
    Returns
    -------
    pd.DataFrame
      A pandas DataFrame with the information of the scope for all 
      conditions
    """
    if not self.__fitted:
      raise Exception("NCA has not been fitted, make sure to .fit() de model")
    return pd.DataFrame(self._scope_, columns=["","Scope"]).set_index("").rename_axis(None, axis=0)
  ## effects_ property
  @property
  def effects_(self):
    """
    Returns
    -------
    pd.DataFrame
      A pandas DataFrame with the information of the effects for all 
      variables and all required ceiling lines
    """
    if not self.__fitted:
      raise Exception("NCA has not been fitted, make sure to .fit() de model")
    return pd.DataFrame(self._effects_)
  ## ceiling_size_ property
  @property
  def ceiling_size_(self):
    """
    Returns
    -------
    pd.DataFrame
      A pandas DataFrame with the information of the ceiling_size for all 
      variables and all required ceiling lines
    """
    if not self.__fitted:
      raise Exception("NCA has not been fitted, make sure to .fit() de model")
    return pd.DataFrame(self._ceiling_size_)
  ## accuracy_ property
  @property
  def accuracy_(self):
    if not self.__fitted:
      raise Exception("NCA has not been fitted, make sure to .fit() de model")
    return pd.DataFrame(self._accuracy_)
  ## condition_inefficiency_ property
  @property
  def condition_inefficiency_(self):
    if not self.__fitted:
      raise Exception("NCA has not been fitted, make sure to .fit() de model")
    return pd.DataFrame(self._condition_inefficiency_)
  ## condition_inefficiency_point_ property
  @property
  def condition_inefficiency_point_(self):
    if not self.__fitted:
      raise Exception("NCA has not been fitted, make sure to .fit() de model")
    return pd.DataFrame(self._condition_inefficiency_point_)
  ## outcome_inefficiency_ property
  @property
  def outcome_inefficiency_(self):
    if not self.__fitted:
      raise Exception("NCA has not been fitted, make sure to .fit() de model")
    return pd.DataFrame(self._outcome_inefficiency_)
  ## outcome_inefficiency_point_ property
  @property
  def outcome_inefficiency_point_(self):
    if not self.__fitted:
      raise Exception("NCA has not been fitted, make sure to .fit() de model")
    return pd.DataFrame(self._outcome_inefficiency_point_)

  # Main methods of the NCA class
  def fit(self, X, y, data):
    '''
    Fits the data to the specified NCA ceilings

    Parameters
    ----------
    X : list
      The list of columns which describe conditions or independent variable(s)
    y : str 
      The column name of the outcome or dependent variable
    data : pandas DataFrame
      A pandas DataFrame object that holds the data to fit the NCA model
    '''

    # Create a hidden attribute for the y and X variable names
    self.__y = y
    self.__X = X
    # Create empty dict to hold effect sizes for all X for selected ceilings
    self._effects_ = {}
    # Create empty dict to hold size of ceiling zone for all X for selected ceilings
    self._ceiling_size_ = {}
    # Create empty dict to hold accuracy for all X for selected ceilings
    self._accuracy_ = {}
    # Create emtpy dict to hold condition inefficiency for all X for selected ceilings
    self._condition_inefficiency_ = {}
    # Create emtpy dict to hold outcome inefficiency for all X for selected ceilings
    self._outcome_inefficiency_ = {}
    # Create emtpy dict to hold condition inefficiency points for all x for selected ceilings
    self._condition_inefficiency_point_ = {}
    # Create emtpy dict to hold outcome inefficiency points for all x for selected ceilings
    self._outcome_inefficiency_point_ = {}

    # Check that the data has the correct shape
    NCA.__check_data(X, y, data)

    # Calculate initial parameters
    self.__scope_limits = NCA.__define_scope_lims(data, X, y)
    # Calculate Scope for all conditions in the dataset
    self._scope_ = [[x,v['Scope']] for (x,v) in self.__scope_limits.items()]

    # Calculate sorted arrays
    self.__sorted_arrays = NCA.__create_sorted_arrays(data, X, y)

    # Calculate CE-FDH if CE-FDH or CR-FDH is in required ceilings
    if "ce-fdh" in self.ceilings or "cr-fdh" in self.ceilings:
      # Calculate CE-FDH envelope list
      self.__CE_FDH_envelope_list_dict = {}
      for x_item in X:
        self.__CE_FDH_envelope_list_dict[x_item] = NCA.__CE_FDH_envelope_list(self.__sorted_arrays[x_item])
      # Calculate CE-FDH peers (Upper left edges)
      self.__CE_FDH_upper_left_edges_dict = {}
      for x_item in X:
        self.__CE_FDH_upper_left_edges_dict[x_item] = NCA.__CE_FDH_peers(self.__CE_FDH_envelope_list_dict[x_item])
      # Calculate CE-FDH effect sizes
      self.__CE_FDH_effect_sizes_dict = {}
      for x_item in X:
        self.__CE_FDH_effect_sizes_dict[x_item] = NCA.__CE_FDH_effect_size(self.__scope_limits[x_item],self.__CE_FDH_upper_left_edges_dict[x_item])[0]
      # Calculate CE-FDH size of ceiling zone
      self.__CE_FDH_ceiling_size_dict = {}
      for x_item in X:
        self.__CE_FDH_ceiling_size_dict[x_item] = NCA.__CE_FDH_effect_size(self.__scope_limits[x_item],self.__CE_FDH_upper_left_edges_dict[x_item])[1]
      # Calculate CE-FDH accuracy (by definition is always 1)
      self.__CE_FDH_accuracy_dict = {}
      for x_item in X:
        self.__CE_FDH_accuracy_dict[x_item] = 1
      # Append reports only in CE-FDH is in the required ceilings
      if "ce-fdh" in self.ceilings:
        ## Append CE-FDH effect sizes on effect sizes main dict
        self._effects_["ce-fdh"] = self.__CE_FDH_effect_sizes_dict
        ## Append CE-FDH size of ceiling zone on size of ceiling zone main dict
        self._ceiling_size_["ce-fdh"] = self.__CE_FDH_ceiling_size_dict
        ## Append CR-FDH accuarcy of accuracy main dict
        self._accuracy_["ce-fdh"] = self.__CE_FDH_accuracy_dict

    # Calculate CR-FDH if CR-FDH is in required ceilings
    if "cr-fdh" in self.ceilings:
      # Calculate CR-FDH OLS parameters
      self.__CR_FDH_OLS_params_dict = {}
      for x_item in X:
        self.__CR_FDH_OLS_params_dict[x_item] = NCA.__CR_FDH_OLS_params(self.__CE_FDH_upper_left_edges_dict[x_item])
      # Calculate CR-FDH polygon array
      self.__CR_FDH_polygon_array_dict = {}
      for x_item in X:
        self.__CR_FDH_polygon_array_dict[x_item] = NCA.__CR_FDH_polygon_array(self.__scope_limits[x_item],
                                                                self.__CR_FDH_OLS_params_dict[x_item][0],
                                                                self.__CR_FDH_OLS_params_dict[x_item][1])
      # Calculate CR-FDH effect sizes
      self.__CR_FDH_effect_sizes_dict = {}
      for x_item in X:
        self.__CR_FDH_effect_sizes_dict[x_item] = NCA.__CR_FDH_effect_size(self.__scope_limits[x_item]["Scope"],
                                                        self.__CR_FDH_polygon_array_dict[x_item])[0]
      ## Append CR-FDH effect sizes on effect sizes main dict
      self._effects_["cr-fdh"] = self.__CR_FDH_effect_sizes_dict
      # Calculate CR-FDH size of ceiling zone
      self.__CR_FDH_ceiling_size_dict = {}
      for x_item in X:
        self.__CR_FDH_ceiling_size_dict[x_item] = NCA.__CR_FDH_effect_size(self.__scope_limits[x_item]["Scope"],
                                                        self.__CR_FDH_polygon_array_dict[x_item])[1]
      ## Append CR-FDH size of ceiling zone on size of ceiling zone main dict
      self._ceiling_size_["cr-fdh"] = self.__CR_FDH_ceiling_size_dict
      # Calculate CR-FDH accuracy
      self.__CR_FDH_accuracy_dict = {}
      for x_item in X:
        self.__CR_FDH_accuracy_dict[x_item] = NCA.__accuracy_from_abline(self.__sorted_arrays[x_item],
                                                                           self.__CR_FDH_OLS_params_dict[x_item][0],
                                                                           self.__CR_FDH_OLS_params_dict[x_item][1])
      ## Append CR-FDH accuarcy of accuracy main dict
      self._accuracy_["cr-fdh"] = self.__CR_FDH_accuracy_dict
      # Calculate CR-FDH condition inefficiency
      self.__CR_FDH_condition_inefficiency_dict = {}
      self.__CR_FDH_condition_innefficiency_point_dict = {}
      for x_item in X:
        self.__CR_FDH_condition_inefficiency_dict[x_item], self.__CR_FDH_condition_innefficiency_point_dict[x_item] = NCA.__condition_inefficiency_from_abline(scope_lims=self.__scope_limits[x_item],
                                                                                                     slope_abline=self.__CR_FDH_OLS_params_dict[x_item][0],
                                                                                                     intercept_abline=self.__CR_FDH_OLS_params_dict[x_item][1])
      ## Append CR-FDH condition inefficiency  to main dict
      self._condition_inefficiency_["cr-fdh"] = self.__CR_FDH_condition_inefficiency_dict
      self._condition_inefficiency_point_["cr-fdh"] = self.__CR_FDH_condition_innefficiency_point_dict
      # Calculate CR-FDH outcome inefficiency
      self.__CR_FDH_outcome_inefficiency_dict = {}
      self.__CR_FDH_outcome_innefficiency_point_dict = {}
      for x_item in X:
        self.__CR_FDH_outcome_inefficiency_dict[x_item], self.__CR_FDH_outcome_innefficiency_point_dict[x_item] = NCA.__outcome_inefficiency_from_abline(scope_lims=self.__scope_limits[x_item],
                                                                                                 slope_abline=self.__CR_FDH_OLS_params_dict[x_item][0],
                                                                                                 intercept_abline=self.__CR_FDH_OLS_params_dict[x_item][1])
      ## Append CR-FDH outcome inefficiency to main dict
      self._outcome_inefficiency_["cr-fdh"] = self.__CR_FDH_outcome_inefficiency_dict
      self._outcome_inefficiency_point_["cr-fdh"] = self.__CR_FDH_outcome_innefficiency_point_dict

    # Calculate OLS model parameters for all observations (a + b*x)
    if "ols" in self.ceilings:
      # Calculate OLS parameters
      self.__OLS_params_dict = {}
      for x_item in X:
        self.__OLS_params_dict[x_item] = NCA.__OLS_params(self.__sorted_arrays[x_item])

    # If no errors were found, change the fitted flag to true
    self.__fitted = True
    # Return the instance so the user can retreive the state of the instance (after fit)
    return self

  def bottleneck(self, ceiling, bottleneck_type = "percentage"):
    """
    Returns the bottleneck table for a specific ceiling technique and 
    type of bottleneck. You first need to fit() the model before using 
    this method.

    Parameters
    ----------
    ceiling : str
      The type of ceiling line to use for the bottleneck table
    bottleneck_type : str, optional
      The type of bottleneck table to return

    Returns
    -------
    pd.DataFrame
      a pandas DataFrame with the information of the bottleneck table

    """
    if self.__fitted and ceiling=="cr-fdh" and "cr-fdh" in self.ceilings:
      return NCA.__bottleneck_table_abline_ceiling(self.__X,
                                             self.__y,
                                             bottleneck_type = bottleneck_type,
                                             OLS_ceiling_line_dict=self.__CR_FDH_OLS_params_dict,
                                             scope_lims_dict = self.__scope_limits,
                                             n_dec = self.__n_dec)
    elif self.__fitted:
      print("Bottleneck can't be shown for the selected ceiling line")
    else:
      print("Bottlenecks can't be shown. First fit the model to your dataset")

  def plot(self, x):
    """
    Plots the NCA scatterplot for a specific determinant and all ceiling
    lines defined in the fit() method. You first need to fit() the 
    model before using this method.

    Parameters
    ----------
    x : str 
      Column name of the determinant in the provided dataframe which is
      to be plotted
    """

    ## Define plot style and params
    plt.rcParams.update({
        'figure.figsize': (10, 8),  # Set figsize in inches
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.grid': False,
        'lines.linewidth': 1
    })
    # Setup plot limits
    Xmin = self.__scope_limits[x]["Xmin"]
    Xmax = self.__scope_limits[x]["Xmax"]
    Ymin = self.__scope_limits[x]["Ymin"]
    Ymax = self.__scope_limits[x]["Ymax"]
    xseq = np.linspace(Xmin, Xmax, num=100)
    ## Setup figure and axis
    fig, ax = plt.subplots()
    ## Draw limits of the scope
    ax.axvline(x=Xmin, alpha=0.5, color="gray", ls="--")
    ax.axvline(x=Xmax, alpha=0.5, color="gray", ls="--")
    ax.axhline(y=Ymin, alpha=0.5, color="gray", ls="--")
    ax.axhline(y=Ymax, alpha=0.5, color="gray", ls="--")
    # Set plot labels
    ax.set_xlabel(x)
    ax.set_ylabel(self.__y)
    ax.set_title(f"NCA Plot: {x} - {self.__y}")
    ## Plot scatterplot
    ax.plot(self.__sorted_arrays[x][:,0],
            self.__sorted_arrays[x][:,1],
            marker="o", ls='', color='#0000ff', alpha=0.5)
    # Plot OLS Regression line
    if "ols" in self.ceilings:
      ax.plot(xseq,  self.__OLS_params_dict[x][1] + self.__OLS_params_dict[x][0] * xseq,
              marker="", ls='-', color='#00ff00', label='OLS')
    # Plot CE-FDH ceiling envelope line
    if "ce-fdh" in self.ceilings:
      ax.plot(self.__CE_FDH_envelope_list_dict[x][:,0], self.__CE_FDH_envelope_list_dict[x][:,1],
              "r-.",label ='CE-FDH')
      '''
      # Plot CE-FDH upper left edges (hidden as they are only meant to validate upper edges)
      ax.plot(self.__CE_FDH_upper_left_edges_dict[x][:,0],
              self.__CE_FDH_upper_left_edges_dict[x][:,1],
              'cv', label="Upper left edges")
      '''
    # Plot CR-FDH ceiling regression line
    if "cr-fdh" in self.ceilings:
      ax.plot(xseq, self.__CR_FDH_OLS_params_dict[x][1] + self.__CR_FDH_OLS_params_dict[x][0] * xseq,
              marker="", ls="-",color="#ffa500", label='CR-FDH')
      '''
      # Plot CR-FDH Polygon bounds (hidden as they are only meant to validate polygon)
      ax.plot(self.__CR_FDH_polygon_array_dict[x][:,0], self.__CR_FDH_polygon_array_dict[x][:,1],
              marker="+", ls="",color="blue", label="CR-FDH Polygon bounds")
      # Plot CR-FDH Neccesity inefficiency boundaries (hidden as they are only meant to validate points)
      ax.plot(self.__CR_FDH_condition_innefficiency_point_dict[x][0],
              self.__CR_FDH_condition_innefficiency_point_dict[x][1],
              marker = "+", ls="", color="orange", label="X_Cmax_CR_FDH")
      ax.plot(self.__CR_FDH_outcome_innefficiency_point_dict[x][0],
              self.__CR_FDH_outcome_innefficiency_point_dict[x][1],
              marker = "+", ls="", color="gray", label="Y_Cmin_CR_FDH")
      '''
    ## Set limits on x and y axis
    margin_size = 0.03
    ax.set_ylim(bottom=Ymin-(Ymax-Ymin)*margin_size, top=Ymax+(Ymax-Ymin)*margin_size)
    ax.set_xlim(left=Xmin-(Xmax-Xmin)*margin_size, right=Xmax+(Xmax-Xmin)*margin_size)
    # Print legend
    ax.legend(loc='upper left')
    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # Show plot
    plt.show()