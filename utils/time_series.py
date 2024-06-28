from .dataset_utils import *


class TimeSeries:
    def __init__(self, csv_path: str, datetime_name: str, datetime_format: str, value_name: str, split_ratio=None,
                 kwargs_features=None, kwargs_lags=None, kwargs_last_n=None, kwargs_prepare_future=None,
                 verbose=True):
        """
        Initializes a TimeSeries object, used for managing time-series data to pass into the
        MasterModel framework (for machine learning models).

        :param csv_path:            path to the .csv file containing the data.
        :param datetime_name:       name of the datetime column (str).
        :param datetime_format:     format of the datetime column (str) (e.g. "%Y-%m-%d %H:%M:%S").
        :param value_name:          name of the value column (str).
        :param kwargs_features:     (kwargs) for initializing the time-series feature creation.
        :param kwargs_lags:         (kwargs) for initializing the time-series lag creation.
        :param verbose:             prints debugging information.
        """
        self.verbose = verbose

        # Initializing the "raw" dataset information:
        self.df_raw = data_read_csv(csv_path, verbose=self.verbose)     # raw DataFrame
        self.datetime_name = datetime_name                              # name of DateTime column
        self.datetime_format = datetime_format                          # format of DateTime column
        self.value_name = value_name                                    # name of Target column

        # Initializing augmentation kwargs:
        self.split_ratio = split_ratio
        self.kwargs_prepare_future = kwargs_prepare_future
        self.kwargs_features = kwargs_features
        self.kwargs_lags = kwargs_lags
        self.kwargs_last_n = kwargs_last_n

        # Initializing modified DataFrame information:
        self.df_augmented = None                # sorted dataframe with DateTime, Target, features, lags
        self.df_augmented_values_only = None    # sorted dataframe (by DateTime) but with only Target
        #   DataFrame with last "n" entries of df_augmented:
        self.df_last_n = None
        #   Features of dataframe:
        self.features = None                    # list of non-lag features (as column names)
        self.lags = None                        # list of lag features (as column names)
        #   (Minimum) lag value used (stored for when predictions are made):
        self.lag_min = None
        self.lag_max = None
        # Immediately try to get the max_lag and min_lag for faster prepare_for_forecast function:
        try:
            self.lag_max = kwargs_lags["lag_base"] * max(kwargs_lags["lag_multiples"])
            self.lag_min = kwargs_lags["lag_base"] * min(kwargs_lags["lag_multiples"])
        except KeyError as e:
            print(f"SoftWarn: Unable to immediately find min/max lags. May slow forecasting. \n"
                  f"Error: {e}.")

        # Initializing the train/test/valid split dataframes:
        self.df_split_train = None
        self.df_split_train_last_n = None   # last "n" entries of training split
        self.df_split_test = None
        self.df_split_test_last_n = None    # last "n" entries of testing split
        self.df_split_valid = None
        self.df_split_valid_last_n = None   # last "n" entries of validation split

        # Initializing future (only) and future (appended) dataframes:
        self.df_future_only = None
        self.df_future_concat = None

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
            df_split_train:         {None if (self.df_split_train is None) else self.df_split_train.shape}
            df_split_test:          {None if (self.df_split_test is None) else self.df_split_test.shape}
            df_split_valid:         {None if (self.df_split_valid is None) else self.df_split_valid.shape}
            
            latest df_raw DateTime:        {self.df_raw[self.datetime_name].iloc[-1]}
        '''
        return string

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
            print(f"Override is set to False, and an existing augmented DataFrame exists.")
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
            df_augmented, features = data_create_features(df_augmented,
                                                          datetime_name=self.datetime_name,
                                                          value_name=self.value_name,
                                                          verbose=self.verbose,
                                                          **kwargs_features)
        #   else, resort to default feature creation:
        else:
            df_augmented, features = data_create_features(df_augmented,
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
            return df_augmented, features, lags, lag_min
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
            split_ratio = self.split_ratio

        # Splitting the augmented dataframe:
        train, test, valid = data_split_train_test_valid(self.df_augmented,
                                                         ratio=split_ratio,
                                                         verbose=self.verbose)
        # Storing the split ratio and dataframes:
        self.split_ratio = split_ratio
        self.df_split_train = train
        self.df_split_test = test
        self.df_split_valid = valid
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
            self.df_future_only = df_future_only
            self.df_future_concat = df_future_concat
            return df_future_only, df_future_concat
        else:
            return df_future_only, df_future_concat

    def df_create_last_n(self, kwargs_last_n=None):
        # Defaulting to initialized kwargs_last_n if not provided:
        if kwargs_last_n is None:
            kwargs_last_n = self.kwargs_last_n
        # Setting df_last_n, df_split_train_last_n, df_split_test_last_n:
        #   note: data_get_last_n() already checks if DataFrame is None.
        #         If so, it simply returns None.
        self.df_last_n = data_get_last_n(dataframe=self.df_augmented, verbose=self.verbose,
                                         **kwargs_last_n)
        self.df_split_train_last_n = data_get_last_n(dataframe=self.df_split_train, verbose=self.verbose,
                                                     **kwargs_last_n)
        self.df_split_test_last_n = data_get_last_n(dataframe=self.df_split_test, verbose=self.verbose,
                                                    **kwargs_last_n)
        self.df_split_valid_last_n = data_get_last_n(dataframe=self.df_split_valid, verbose=self.verbose,
                                                     **kwargs_last_n)
        return

    def prepare_from_scratch(self, kwargs_last_n=None, split_ratio=None, kwargs_features=None,
                             kwargs_lags=None, kwargs_prepare_future=None):
        """
        Prepares the entire instance from scratch. Overrides any existing values automatically.


        :param kwargs_last_n:           parameters to add last "n" entries to datasets.
        :param split_ratio:             a list of three floats (adding to 1) representing train/test/valid ratios.
        :param kwargs_features:         time-series features to add (in the form passed to df_augment()).
        :param kwargs_lags:             lag parameters to add (in the form passed to df_augment()).
        :param kwargs_prepare_future:   arguments for preparation of future (window_size/step_size).
        :return:                        None.
        """
        # If split_ratio, kwargs_features and kwargs_lags are not provided, default to whatever
        #   was used to initialize the TimeSeries:
        if kwargs_last_n is None:
            kwargs_last_n = self.kwargs_last_n
        if split_ratio is None:
            split_ratio = self.split_ratio
        if kwargs_features is None:
            kwargs_features = self.kwargs_features
        if kwargs_lags is None:
            kwargs_lags = self.kwargs_lags
        if kwargs_prepare_future is None:
            kwargs_prepare_future = self.kwargs_prepare_future

        # Augment and store the raw data:
        self.df_augment(override=True, kwargs_features=kwargs_features, kwargs_lags=kwargs_lags)
        # Split and store the raw data into train/test/valid datasets:
        self.df_split_ttv(split_ratio, override=True)
        # Create and store future (both "only" and "concat") datasets:
        self.df_create_future(**kwargs_prepare_future,
                              kwargs_features=kwargs_features, kwargs_lags=kwargs_lags)
        # Create last_n entries:
        self.df_create_last_n(kwargs_last_n=kwargs_last_n)
        return

    def prepare_for_forecast(self, kwargs_last_n=None, split_ratio=None, kwargs_features=None,
                             kwargs_lags=None, kwargs_prepare_future=None):
        """
        Prepares the TimeSeries object for forecasting, skipping any unnecessary steps (such as
        data train/test/valid splitting), and including optimizations (such as dataset size optimization)
        to speed up critical augmentation processes.

        Consumes the same kwargs as passed in prepare_from_scratch. Does not use kwargs_last_n or
        split_ratio.

        :param kwargs_last_n:           unused, present for standardization purposes.
        :param split_ratio:             unused, present for standardization purposes.
        :param kwargs_features:         features as kwargs.
        :param kwargs_lags:             lags as kwargs.
        :param kwargs_prepare_future:   future dataset preparation as kwargs.
        :return:                        None.
        """
        if kwargs_features is None:
            kwargs_features = self.kwargs_features
        if kwargs_lags is None:
            kwargs_lags = self.kwargs_lags
        if kwargs_prepare_future is None:
            kwargs_prepare_future = self.kwargs_prepare_future

        # Only consider as many entries as necessary:
        if self.lag_max is not None:
            n_entries = max(self.lag_max, 500)  # NOTE: 500 needs to be __lookback.
            # Getting last entries of df_raw AFTER sorting first:
            self.df_raw = data_datetime_conversion(self.df_raw, self.datetime_name, self.datetime_format,
                                                   verbose=self.verbose)
            self.df_raw = data_datetime_sort(self.df_raw, self.datetime_name, verbose=self.verbose)
            self.df_raw = self.df_raw.iloc[-n_entries:]

        # Augment and store the raw data:
        self.df_augment(update_self=True, override=True, kwargs_features=kwargs_features,
                        kwargs_lags=kwargs_lags)
        # Create and store future (both "only" and "concat") datasets:
        self.df_create_future(**kwargs_prepare_future,
                              kwargs_features=kwargs_features, kwargs_lags=kwargs_lags)
        return
