export class AppError extends Error {
  constructor(code, message, options = {}) {
    super(message);
    this.name = "AppError";
    this.code = code;
    this.statusCode = options.statusCode ?? 400;
    this.details = options.details ?? undefined;
  }
}

export function createError(code, options = {}) {
  const catalog = {
    INVALID_INPUT: { message: "The submitted payload is invalid.", statusCode: 400 },
    INVALID_TEAM: { message: "Team details are invalid.", statusCode: 400 },
    TEAM_LIMIT_REACHED: { message: "The maximum number of teams has been reached.", statusCode: 409 },
    ROUND_NOT_ACTIVE: { message: "The round is not currently accepting decisions.", statusCode: 409 },
    ROUND_CLOSED: { message: "The round is closed.", statusCode: 409 },
    ROUND_ALREADY_PAUSED: { message: "The round is already paused.", statusCode: 409 },
    ROUND_NOT_PAUSED: { message: "The round is not paused.", statusCode: 409 },
    ROUND_ALREADY_EVALUATED: { message: "This round has already been evaluated.", statusCode: 409 },
    ALREADY_SUBMITTED: { message: "This team has already submitted a decision for the round.", statusCode: 409 },
    GAME_ALREADY_STARTED: { message: "The game has already started.", statusCode: 409 },
    GAME_NOT_STARTED: { message: "The game has not started yet.", statusCode: 409 },
    GAME_FINISHED: { message: "The game has already finished.", statusCode: 409 },
    RESOURCE_BUSY: { message: "The game state is busy. Please retry.", statusCode: 423 },
    AUTH_REQUIRED: { message: "Admin authentication is required.", statusCode: 401 },
    INVALID_ADMIN_SECRET: { message: "The provided admin secret is invalid.", statusCode: 401 },
    SERVER_SHUTTING_DOWN: { message: "The server is shutting down.", statusCode: 503 },
    INTERNAL_ERROR: { message: "An unexpected error occurred.", statusCode: 500 },
  };

  const entry = catalog[code] ?? catalog.INTERNAL_ERROR;
  return new AppError(code, options.message ?? entry.message, {
    statusCode: options.statusCode ?? entry.statusCode,
    details: options.details,
  });
}

export function isAppError(error) {
  return error instanceof AppError;
}

export function serializeError(error) {
  if (isAppError(error)) {
    return {
      code: error.code,
      message: error.message,
      ...(error.details ? { details: error.details } : {}),
    };
  }

  return {
    code: "INTERNAL_ERROR",
    message: "An unexpected error occurred.",
  };
}
