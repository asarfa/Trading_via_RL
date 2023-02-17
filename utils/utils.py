from datetime import datetime, timedelta


def split_dates(split: float = None, start_date: datetime = None, end_date: datetime = None,
                hour_start: float = 9.5, hour_end: float = 15.5):
    start_train_date = start_date
    end_test_date = end_date
    end_train_date = (start_train_date + (end_test_date - start_train_date) * split).replace(hour=0)
    start_test_date = end_train_date + timedelta(hours=24)
    end_train_date += timedelta(hours=hour_end)
    start_test_date += timedelta(hours=hour_start)
    return [start_train_date, end_train_date, start_test_date, end_test_date]
