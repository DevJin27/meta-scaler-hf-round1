import pino from "pino";

export function createLogger(config) {
  return pino({
    level: config.logLevel,
    base: {
      service: "signal-game-backend",
      env: config.env,
    },
    redact: {
      paths: ["req.headers.authorization", "adminSecret"],
      remove: true,
    },
    timestamp: pino.stdTimeFunctions.isoTime,
  });
}
