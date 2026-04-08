export class LifecycleService {
  constructor({ engine, config, logger, onEvent }) {
    this.engine = engine;
    this.config = config;
    this.logger = logger;
    this.onEvent = onEvent;
    this.timer = null;
    this.running = false;
  }

  start() {
    if (this.timer) {
      return;
    }

    this.timer = setInterval(async () => {
      if (this.running) {
        return;
      }

      this.running = true;

      try {
        const evaluation = await this.engine.evaluateExpiredRound();
        if (evaluation) {
          await this.onEvent({ type: "round:evaluated", payload: evaluation });
        }

        const followUp = await this.engine.autoAdvanceIfReady();
        if (followUp) {
          await this.onEvent(followUp);
        }
      } catch (error) {
        this.logger.error({ err: error }, "lifecycle tick failed");
      } finally {
        this.running = false;
      }
    }, this.config.lifecycleTickMs);

    this.timer.unref?.();
  }

  stop() {
    if (!this.timer) {
      return;
    }

    clearInterval(this.timer);
    this.timer = null;
  }
}
