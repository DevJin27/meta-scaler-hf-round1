import express from "express";

import { isAppError, serializeError } from "./errors.js";

export function asyncRoute(handler) {
  return async (req, res, next) => {
    try {
      await handler(req, res, next);
    } catch (error) {
      next(error);
    }
  };
}

export function createBaseApp({ logger, allowedOrigins }) {
  const app = express();
  app.set("trust proxy", true);
  app.use(express.json({ limit: "256kb" }));
  app.use(express.urlencoded({ extended: false }));

  app.use((req, res, next) => {
    const requestLogger = logger.child({
      requestId: req.headers["x-request-id"] || undefined,
      method: req.method,
      path: req.path,
    });

    const startedAt = Date.now();
    req.log = requestLogger;

    res.on("finish", () => {
      requestLogger.info(
        {
          statusCode: res.statusCode,
          durationMs: Date.now() - startedAt,
          origin: req.headers.origin || null,
        },
        "http request completed",
      );
    });

    const origin = req.headers.origin;
    if (!origin || allowedOrigins.includes("*") || allowedOrigins.includes(origin)) {
      if (allowedOrigins.includes("*")) {
        res.header("Access-Control-Allow-Origin", "*");
      } else if (origin) {
        res.header("Access-Control-Allow-Origin", origin);
        res.header("Vary", "Origin");
      }

      res.header("Access-Control-Allow-Headers", "Content-Type, X-Admin-Secret, X-Request-Id");
      res.header("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
      if (req.method === "OPTIONS") {
        res.status(204).end();
        return;
      }

      next();
      return;
    }

    res.status(403).json({
      ok: false,
      error: {
        code: "FORBIDDEN_ORIGIN",
        message: "Origin is not allowed.",
      },
    });
  });

  return app;
}

export function attachErrorHandler(app, logger) {
  app.use((error, req, res, _next) => {
    const statusCode = isAppError(error) ? error.statusCode : 500;
    const payload = serializeError(error);
    const log = req?.log ?? logger;

    log.error(
      {
        err: error,
        statusCode,
        payload,
      },
      "request failed",
    );

    res.status(statusCode).json({ ok: false, error: payload });
  });
}
