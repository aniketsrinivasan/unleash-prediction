from .dataset_utils_deprecated import *


class TimeSeries:
    def __init__(self, csv_path: str, datetime_name: str, datetime_format: str,
                 value_name: str, verbose=True):
        """
        Initializes a TimeSeries object, used for managing time-series data to pass into model_framework.

        :param csv_path:            path to the .csv file containing the data.
        :param datetime_name:       name of the datetime column (str).
        :param datetime_format:     format of the datetime column (str) (e.g. "%Y-%m-%d %H:%M:%S").
        :param value_name:          name of the value column (str).
        :param verbose:             prints debugging information.
        """

        self.verbose = verbose

        # Initializing the "raw" dataframe (read directly from the .csv):
        self.df_raw = data_read_csv(csv_path, verbose=self.verbose)
        self.datetime_name = datetime_name
        self.datetime_format = datetime_format
        self.value_name = value_name

        # Initializing modified dataframes:
        #   Dataframe after passing data_augment:
        self.df_augmented = None                # sorted dataframe with DateTime, Target, features, lags
        self.df_augmented_values_only = None    # sorted dataframe (by DateTime) but with only Target
        #   Features of dataframe:
        self.features = None                # list of non-lag features (as column names)
        self.lags = None                    # list of lag features (as column names)
        #   (Minimum) lag value used (stored for when predictions are made):
        self.lag_min = None

        # Initializing the train/test/valid split dataframes:
        self.split_ratio = None
        self.df_split_train = None
        self.df_split_test = None
        self.df_split_valid = None

        # Initializing future (only) and future (appended) dataframes:
        self.df_future_only = None
        self.df_future_concat = None

        # Initializing prediction dataframe(s) and information:
        self.df_future_predictions = None       # DataFrame containing past info and future predictions
        self.isfuture_name = None               # name of mask column isFuture (as column name)

    # When the object is called using print() or otherwise (representation).
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
            lags:                   {self.lags}
            lag_min:                {self.lag_min} (entries)
            
            split_ratio:            {self.split_ratio} (train/test/valid)
            df_split_train:         {self.df_split_train.shape}
            df_split_test:          {self.df_split_test.shape}
            df_split_valid:         {self.df_split_valid.shape}
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
    # NOTE: implement interpolation methods during initial augmentation for missing values?
    #       (find some way to deal w/ missing values basically).
    def df_augment(self, custom_df=None, update_self=True, override=False,
                   kwargs_features=None, kwargs_lags=None):
        """
        Augments the dataframe in self.df_raw (by default, unless custom_df is provided) using the
        provided kwargs_features and kwargs_lags. Either updates the instance, or returns a tuple.

        :param custom_df:           if a different pd.DataFrame (other than self.df_raw) is to be augmented.
                                        requires:   has same column names as self.df_raw [not asserted]
        :param update_self:         whether to update current object instance.
        :param override:            whether to override existing instance information.
        :param kwargs_features:     features provided for data_datetime_create_features(...).
        :param kwargs_lags:         lags provided for data_create_lags(...).
        :return:                    either nothing, or (if update_self==False), tuple of:
                                        df_augmented, features, lags, minimum_lag
        """

        # If the data has already been augmented, do not augment again if override==False:
        if (self.df_augmented is not None) and (override is False) and (update_self is True):
            print(f"Override is set to False, and an existing augmented dataframe exists.")
            print(f"Aborting data augmentation.")
            return

        # If custom_df is not provided, we default to using self.df_raw:
        if custom_df is None:
            # Initializing df_augmented to only contain the DateTime and Target columns from
            #   the raw data:
            df_augmented = self.df_raw[[self.value_name, self.datetime_name]].copy()
        else:
            df_augmented = custom_df[[self.value_name, self.datetime_name]].copy()

        # Converting self.datetime_name to a DateTime column:
        df_augmented = data_datetime_conversion(df_augmented, self.datetime_name,
                                                self.datetime_format, self.verbose)
        # Sorting by DateTime:
        df_augmented = data_datetime_sort(df_augmented, self.datetime_name, self.verbose)

        # Creating time-series features (e.g. "Week", "Month", "Year"):
        #   If kwargs_features are provided, use them:
        if kwargs_features is not None:
            df_augmented, features = data_datetime_create_features(df_augmented,
                                                                   datetime_name=self.datetime_name,
                                                                   value_name=self.value_name,
                                                                   verbose=self.verbose,
                                                                   **kwargs_features)
        #   else, resort to default feature creation:
        else:
            df_augmented, features = data_datetime_create_features(df_augmented,
                                                                   datetime_name=self.datetime_name,
                                                                   value_name=self.value_name,
                                                                   verbose=self.verbose)

        # Creating time-series lags (e.g. "lag_1w", "lag_4w", ...):
        #   If kwargs_lags are provided, use them:
        if kwargs_lags is not None:
            df_augmented, lag_min, lags = data_create_lags(dataframe=df_augmented,
                                                           value_name=self.value_name,
                                                           verbose=self.verbose,
                                                           **kwargs_lags)
        #   else, create and use a "default" lags:
        else:
            default_lags = dict(lag_base=1,
                                lag_multiples=[],
                                lag_label="")
            df_augmented, lag_min, lags = data_create_lags(dataframe=df_augmented,
                                                           value_name=self.value_name,
                                                           verbose=self.verbose,
                                                           **default_lags)

        # Storing all the information from this process:
        if update_self:
            self.df_augmented = df_augmented.copy()
            self.df_augmented_values_only = df_augmented[self.value_name].copy()
            self.features = list(features.values())
            self.lags = lags
            self.lag_min = lag_min
            return
        else:
            return df_augmented, features, lags, lag_min

    def df_split_ttv(self, split_ratio=None, override=False):
        """
        Splits self.df_augmented into three pd.DataFrames: train, test, valid. Does not override any
        existing values by default.

        :param split_ratio:         a list of three floats (adding to 1) representing train/test/valid ratios.
        :param override:            whether to override existing values in the instance.
        :return:                    None.
        """

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

    def df_create_future(self, window_size: int, step_size: str,
                         kwargs_features=None, kwargs_lags=None, update_self=True):
        """
        Creates a "future" for the dataset in self.df_raw by adding window_size entries stepping by step_size,
        augmenting the dataset (adding features and lags), and either storing or returning two datasets:
        one concatenated with past values, and another not.

        :param window_size:             number of entries to add to the future.
        :param step_size:               the step (as a str) between each entry (e.g. "1h").
        :param kwargs_features:         kwargs to pass to df_augment().
        :param kwargs_lags:             kwargs to pass to df_augment().
        :param update_self:             whether to update instance (if False, then datasets are returned).
        :return:                        Optional[tuple[df_future_only, df_future_concat]]
        """

        # Creating df_raw_sorted (pd.DataFrame) which is a sorted version of the raw data:
        df_raw_sorted = self.df_raw[[self.value_name, self.datetime_name]].copy()
        df_raw_sorted = data_datetime_conversion(df_raw_sorted, self.datetime_name,
                                                 self.datetime_format, self.verbose)
        df_raw_sorted = data_datetime_sort(df_raw_sorted, self.datetime_name, self.verbose)

        # Creating a future window of df_raw_sorted which is appended:
        df_future_concat = data_create_future(dataframe=df_raw_sorted,
                                              datetime_name=self.datetime_name,
                                              value_name=self.value_name,
                                              window_size=window_size,
                                              step=step_size,
                                              append=True,
                                              verbose=self.verbose)

        # Creating features and lags on df_future_concat:
        df_future_concat, features, lags, lag_min = self.df_augment(custom_df=df_future_concat,
                                                                    update_self=False,
                                                                    override=False,
                                                                    kwargs_features=kwargs_features,
                                                                    kwargs_lags=kwargs_lags)
        # Creating a future-only dataframe:
        df_future_only = df_future_concat.copy().iloc[-window_size:]

        # Updating and/or returning as required:
        if update_self:
            self.df_future_only = df_future_only.copy()
            self.df_future_concat = df_future_concat.copy()
            return
        else:
            return df_future_only, df_future_concat

    def prepare_from_scratch(self, future_window_size: int, future_step_size: str,
                             split_ratio=None, kwargs_features=None, kwargs_lags=None):
        """
        Prepares the entire instance from scratch. Overrides any existing values automatically.

        :param future_window_size:      number of entries to add to the future.
        :param future_step_size:        the step (as a str) between each entry (e.g. "1h").
        :param split_ratio:             a list of three floats (adding to 1) representing train/test/valid ratios.
        :param kwargs_features:         time-series features to add (in the form passed to df_augment()).
        :param kwargs_lags:             lag parameters to add (in the form passed to df_augment()).
        :return:                        None.
        """
        # Augment and store the raw data:
        self.df_augment(override=True, kwargs_features=kwargs_features, kwargs_lags=kwargs_lags)
        # Split and store the raw data into train/test/valid datasets:
        self.df_split_ttv(split_ratio, override=True)
        # Create and store future (both "only" and "concat") datasets:
        self.df_create_future(window_size=future_window_size, step_size=future_step_size,
                              kwargs_features=kwargs_features, kwargs_lags=kwargs_lags)
        return
