import time


class IntervalSummaryWriter:
    """
    Summary writer wrapper designed to reduce the size of tf.events files.
    It will prevent the learner from writing the summaries more often than a specified interval, i.e. if the
    current interval is 20 seconds and we wrote our last summary for a particular summary key at 01:00, all summaries
    until 01:20 for that key will be ignored.

    The interval is adaptive: it will approach 1/200th of the total training time, but no less than interval_sec_min
    and no greater than interval_sec_max.

    This was created to facilitate really big training runs, such as with Population-Based training, where summary
    folders reached tens of gigabytes.
    """

    def __init__(self, summary_writer, cfg, time_fn=None):
        self.time_fn = time_fn or time.time
        self.experiment_start = self.time_fn()

        # prevents noisy summaries when experiments are restarted
        self.defer_summaries_sec = cfg.get('defer_summaries_sec', 5)

        self.interval_sec_min = cfg.get('summaries_interval_sec_min', 5)
        self.interval_sec_max = cfg.get('summaries_interval_sec_max', 300)
        self.last_interval = self.interval_sec_min

        # interval between summaries will be close to this fraction of the total training time,
        # i.e. for a run that lasted 200 minutes we write one summary every minute.
        self.summaries_relative_step = 1.0 / 200

        self.writer = summary_writer
        self.last_write_for_tag = dict()

    def _calc_interval(self, now):
        """Write summaries more often in the beginning of the run."""
        if self.last_interval >= self.interval_sec_max:
            return self.last_interval

        seconds_since_start = now - self.experiment_start
        interval = seconds_since_start * self.summaries_relative_step
        interval = min(interval, self.interval_sec_max)
        interval = max(interval, self.interval_sec_min)
        self.last_interval = interval

        return interval

    def _should_write(self, tag, step):
        if step == 0:
            return False

        now = self.time_fn()
        seconds_since_start = now - self.experiment_start
        if seconds_since_start < self.defer_summaries_sec:
            return False

        last_write = self.last_write_for_tag.get(tag, 0)
        seconds_since_last_write = now - last_write
        interval = self._calc_interval(now)
        if seconds_since_last_write >= interval:
            self.last_write_for_tag[tag] = now
            return True
        return False

    def add_scalar(self, tag, value, step, *args, **kwargs):
        if self._should_write(tag, step):
            self.writer.add_scalar(tag, value, step, *args, **kwargs)

    def add_histogram(self, tag, values, step, *args, **kwargs):
        if self._should_write(tag, step):
            self.writer.add_histogram(tag, values, step, *args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.writer, attr)
