from .dataset_utils import *


class TimeSeries:
    def __init__(self, csv_path: str, datetime_name: str, datetime_format: str,
                 value_name: str, verbose=True):
        self.verbose = verbose

        # Initializing the "raw" dataframe (read directly from the .csv):
        self.df_raw = data_read_csv(csv_path, verbose=self.verbose)
        self.datetime_name = datetime_name
        self.datetime_format = datetime_format
        self.value_name = value_name

        # Initializing modified dataframes:
        #   Dataframe after passing data_augment:
        self.df_augmented = None            # sorted dataframe with DateTime, Target, features, lags
        #   Features of dataframe:
        self.features = None                # list of non-lag features (as column names)
        self.lags_multiples = None          # list of lag multiples (as integers)
        self.lag_base = None                # the lag base (as number of entries)
        #   (Minimum) lag value used (stored for when predictions are made):
        self.lag_min = None

        # Initializing the train/test/valid split dataframes:
        self.split_ratio = None
        self.df_split_train = None
        self.df_split_test = None
        self.df_split_valid = None

    def __repr__(self):
        string = f'''
        TimeSeries object with the following attributes:
            df_raw (shape):         {self.df_raw.shape}      
        
            verbose:                {self.verbose}
            datetime_name:          {self.datetime_name}
            datetime_format:        {self.datetime_format}
            value_name:             {self.value_name}
            
            df_augmented (shape):   {self.df_augmented.shape if self.df_augmented is not None 
                                                             else "not created."}
            features:               {self.features}
            lag_min:                {self.lag_min}
        '''
        return string

    # data_augment(self, kwargs) augments the dataframe in self.df_raw and creates a new instance
    #   attribute self.df_augmented.
    # The following augmentations are currently executed:
    #   * all columns that are not self.datetime_name or self.value_name are removed
    #   * the self.datetime_name column is converted to a DateTime format
    #   * the dataframe is sorted by DateTime
    #   * time-series features are created (based on kwargs_features)
    #   * time-series lags are created (based on kwargs_lags)
    def df_augment(self, override=False, kwargs_features=None, kwargs_lags=None):
        # NOTE: modify such that self.lag_multiples and self.lag_base is updated as well.
        #       may need to change implementation of data_create_lags.

        # If the data has already been augmented, do not augment again if override==False:
        if (self.df_augmented is not None) and (override is False):
            print(f"Override is set to False, and an existing augmented dataframe exists.")
            print(f"Aborting data augmentation.")
            return

        # Initializing df_augmented to only contain the DateTime and Target columns from
        #   the raw data:
        df_augmented = self.df_raw[[self.value_name, self.datetime_name]].copy()

        # Converting self.datetime_name to a DateTime column:
        if kwargs_lags is None:
            kwargs_lags = {"lag_base": 24 * 7, "lag_multiples": [4, 8, 12, 28, 52]}
        df_augmented = data_datetime_conversion(df_augmented, self.datetime_name,
                                                self.datetime_format, self.verbose)
        # Sorting by DateTime:
        df_augmented = data_datetime_sort(df_augmented, self.datetime_name, self.verbose)

        # Creating time-series features (e.g. "Week", "Month", "Year"):
        #   If kwargs_features are provided, use them:
        if kwargs_features is not None:
            df_augmented, features = data_datetime_create_features(df_augmented,
                                                                   datetime_name=self.datetime_name,
                                                                   verbose=self.verbose,
                                                                   **kwargs_features)
        #   else, resort to default feature creation:
        else:
            df_augmented, features = data_datetime_create_features(df_augmented,
                                                                   datetime_name=self.datetime_name,
                                                                   verbose=self.verbose)

        # Creating time-series lags (e.g. "lag_1w", "lag_4w", ...):
        #   If kwargs_lags are provided, use them:
        if kwargs_lags is not None:
            df_augmented, lag_min = data_create_lags(df_augmented, self.value_name,
                                                     verbose=self.verbose,
                                                     **kwargs_lags)
        #   else, create and use a "default" lags:
        else:
            default_lags = dict(lag_base=24*7,
                                lag_multiples=[4, 8, 16, 54],
                                lag_label="w")
            df_augmented, lag_min = data_create_lags(df_augmented, self.value_name,
                                                     verbose=self.verbose,
                                                     **default_lags)

        # Storing all the information from this process:
        self.df_augmented = df_augmented
        self.features = features
        self.lag_min = lag_min
        return

    def df_split_ttv(self, split_ratio=None, override=False):
        # If the split has already been created and override==False, do not change:
        #   we check whether the split has been created by checking whether an existing split ratio exists.
        if (self.split_ratio is not None) and (override is False):
            print(f"Override is set to {override}, and an existing split exists.")
            print(f"Aborting data split.")
            return

        # If split_ratio is not provided, set to default values below:
        if split_ratio is None:
            split_ratio = [0.7, 0.2, 0.1]

        # Splitting the augmented dataframe:
        train, test, valid = data_split_train_test_valid(self.df_augmented,
                                                         ratio=split_ratio,
                                                         verbose=self.verbose)
        # Storing the split ratio and dataframes:
        self.split_ratio = split_ratio
        self.df_split_train = train.copy()
        self.df_split_test = test.copy()
        self.df_split_valid = valid.copy()
        return

    def df_create_future(self):
        # initialize a dataframe for future only (DateTime and Target)
        # create a dataframe of both future and raw_data[[DateTime, Target]] concatenated
        #   pass through augmentation process
        # extract (as a copy) only the "future" part of the concatenated dataframe
        # store df_future_only (only future entries w/ features/lags) and df_future_concat (concatenated)
        pass
