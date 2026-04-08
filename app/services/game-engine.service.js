import { createError } from "../core/errors.js";
import { SIGNALS } from "../models/signals.js";
import { getSignalForRound, toPublicSignal } from "../models/signals.js";

function sortLeaderboard(entries) {
  return [...entries].sort((left, right) => {
    if (right.score !== left.score) {
      return right.score - left.score;
    }
    return left.name.localeCompare(right.name);
  });
}

function roundPenalty(value) {
  return Math.round(value * 0.65);
}

function evaluateDecision(signal, decision) {
  if (decision === "TRADE" && signal.isAlpha) {
    return {
      verdict: "CORRECT_ALPHA",
      delta: signal.value,
    };
  }

  if (decision === "IGNORE" && !signal.isAlpha) {
    return {
      verdict: "CORRECT_IGNORE",
      delta: 100,
    };
  }

  if (decision === "TRADE" && !signal.isAlpha) {
    return {
      verdict: "WRONG",
      delta: -roundPenalty(signal.value),
    };
  }

  if ((decision === "IGNORE" || decision === null) && signal.isAlpha) {
    return {
      verdict: "MISS_ALPHA",
      delta: -100,
    };
  }

  return {
    verdict: "NO_ACTION",
    delta: 0,
  };
}

export class GameEngineService {
  constructor({ repository, lockService, config, logger, now = () => Date.now() }) {
    this.repository = repository;
    this.lockService = lockService;
    this.config = config;
    this.logger = logger;
    this.now = now;
    this.presenceResolver = async () => 0;
  }

  setPresenceResolver(resolver) {
    this.presenceResolver = resolver;
  }

  assertAdminSecret(secret) {
    if (String(secret || "") !== this.config.adminSecret) {
      throw createError("INVALID_ADMIN_SECRET");
    }
  }

  getRemainingMs(state, now = this.now()) {
    if (state.phase === "paused") {
      return Math.max(0, state.pauseRemainingMs);
    }

    if (state.phase !== "live") {
      return 0;
    }

    return Math.max(0, state.roundEndsAt - now);
  }

  async buildLeaderboard(teams) {
    const withPresence = await Promise.all(
      teams.map(async (team) => {
        const connectedCount = await this.presenceResolver(team.teamId);
        return {
          teamId: team.teamId,
          name: team.name,
          score: team.score,
          connected: connectedCount > 0,
          connectedCount,
          lastDecision: team.lastDecision ?? null,
          lastDelta: team.lastDelta ?? 0,
          lastSubmittedRound: team.lastSubmittedRound ?? 0,
        };
      }),
    );

    return sortLeaderboard(withPresence).map((team, index) => ({
      rank: index + 1,
      ...team,
    }));
  }

  async buildSnapshot({ teamId = null, isAdmin = false } = {}) {
    const [state, teams, lastEvaluation] = await Promise.all([
      this.repository.getState(),
      this.repository.listTeams(),
      this.repository.getLastEvaluation(),
    ]);
    const signal = getSignalForRound(state.round);
    const submissions = state.round > 0 ? await this.repository.getSubmissions(state.round) : {};
    const leaderboard = await this.buildLeaderboard(teams);
    const team = teamId ? teams.find((entry) => entry.teamId === teamId) ?? null : null;
    const viewerSubmission = teamId ? submissions[teamId] ?? null : null;
    const serverTime = this.now();

    const snapshot = {
      phase: state.phase,
      round: state.round,
      totalRounds: state.totalRounds,
      signal: state.phase === "idle" ? null : toPublicSignal(signal),
      timer: {
        durationMs: state.roundDurationMs,
        endsAt: state.roundEndsAt || null,
        remainingMs: this.getRemainingMs(state, serverTime),
        locked: state.inputLocked,
        serverTime,
      },
      submissionCount: Object.keys(submissions).length,
      maxTeams: this.config.maxTeams,
      leaderboard,
      viewer: team
        ? {
            teamId: team.teamId,
            name: team.name,
            score: team.score,
            hasSubmitted: Boolean(viewerSubmission),
            decision: viewerSubmission?.decision ?? null,
          }
        : null,
      lastEvaluation:
        state.phase === "live"
          ? null
          : lastEvaluation,
    };

    if (isAdmin) {
      return {
        ...snapshot,
        admin: {
          submissions,
          state,
        },
      };
    }

    return snapshot;
  }

  async joinTeam({ teamId, name }) {
    return this.lockService.withLock(
      "teams",
      async () => {
        const [existing, teamCount] = await Promise.all([
          this.repository.getTeam(teamId),
          this.repository.countTeams(),
        ]);

        if (!existing && teamCount >= this.config.maxTeams) {
          throw createError("TEAM_LIMIT_REACHED", {
            details: { maxTeams: this.config.maxTeams },
          });
        }

        const now = this.now();
        const team = existing ?? {
          teamId,
          score: 0,
          createdAt: now,
          lastDecision: null,
          lastDelta: 0,
          lastSubmittedRound: 0,
        };

        const nextTeam = {
          ...team,
          name,
          updatedAt: now,
        };

        await this.repository.saveTeam(nextTeam);

        return {
          team: nextTeam,
          snapshot: await this.buildSnapshot({ teamId }),
        };
      },
      { attempts: 5, retryDelayMs: 25 },
    );
  }

  async startGame() {
    return this.lockService.withLock(
      "game-admin",
      async () => {
        const now = this.now();
        const state = await this.repository.getState();

        if (state.phase === "live" || state.phase === "paused") {
          throw createError("GAME_ALREADY_STARTED");
        }

        const teams = await this.repository.listTeams();
        const resetTeams = teams.map((team) => ({
          ...team,
          score: 0,
          lastDecision: null,
          lastDelta: 0,
          lastSubmittedRound: 0,
          updatedAt: now,
        }));

        await Promise.all([
          this.repository.saveTeams(resetTeams),
          this.repository.clearLastEvaluation(),
          this.repository.clearAllRoundSubmissions(),
          this.repository.saveState({
            phase: "live",
            round: 1,
            totalRounds: state.totalRounds,
            currentSignalId: getSignalForRound(1)?.id ?? "",
            roundDurationMs: this.config.roundDurationMs,
            roundStartedAt: now,
            roundEndsAt: now + this.config.roundDurationMs,
            inputLocked: false,
            pauseRemainingMs: 0,
            autoAdvanceAt: 0,
            lastUpdatedAt: now,
          }),
        ]);

        return this.buildRoundStartedPayload(1, now);
      },
      { attempts: 5, retryDelayMs: 25 },
    );
  }

  async resetGame() {
    return this.lockService.withLock(
      "game-admin",
      async () => {
        const now = this.now();
        const state = await this.repository.getState();
        const teams = await this.repository.listTeams();
        const resetTeams = teams.map((team) => ({
          ...team,
          score: 0,
          lastDecision: null,
          lastDelta: 0,
          lastSubmittedRound: 0,
          updatedAt: now,
        }));

        await Promise.all([
          this.repository.saveTeams(resetTeams),
          this.repository.clearLastEvaluation(),
          this.repository.clearAllRoundSubmissions(),
          this.repository.saveState({
            phase: "idle",
            round: 0,
            totalRounds: state.totalRounds,
            currentSignalId: "",
            roundDurationMs: this.config.roundDurationMs,
            roundStartedAt: 0,
            roundEndsAt: 0,
            inputLocked: true,
            pauseRemainingMs: 0,
            autoAdvanceAt: 0,
            lastUpdatedAt: now,
          }),
        ]);

        return this.buildSnapshot();
      },
      { attempts: 5, retryDelayMs: 25 },
    );
  }

  async submitDecision({ teamId, decision, round }) {
    const [team, state] = await Promise.all([
      this.repository.getTeam(teamId),
      this.repository.getState(),
    ]);

    if (!team) {
      throw createError("INVALID_TEAM", {
        details: { teamId },
      });
    }

    const currentRound = round ?? state.round;
    const submittedAt = this.now();
    const result = await this.repository.submitDecision({
      teamId,
      round: currentRound,
      decision,
      submittedAt,
    });

    if (result !== "OK") {
      throw createError(result);
    }

    return {
      accepted: true,
      teamId,
      round: currentRound,
      decision,
      submittedAt,
    };
  }

  async pauseRound() {
    return this.lockService.withLock("game-admin", async () => {
      const now = this.now();
      const state = await this.repository.getState();

      if (state.phase === "paused") {
        throw createError("ROUND_ALREADY_PAUSED");
      }

      if (state.phase !== "live") {
        throw createError("ROUND_NOT_ACTIVE");
      }

      const pauseRemainingMs = Math.max(0, state.roundEndsAt - now);
      const nextState = {
        ...state,
        phase: "paused",
        pauseRemainingMs,
        inputLocked: true,
        lastUpdatedAt: now,
      };

      await this.repository.saveState(nextState);

      return {
        round: state.round,
        pausedAt: now,
        remainingMs: pauseRemainingMs,
      };
    });
  }

  async resumeRound() {
    return this.lockService.withLock("game-admin", async () => {
      const now = this.now();
      const state = await this.repository.getState();

      if (state.phase !== "paused") {
        throw createError("ROUND_NOT_PAUSED");
      }

      const nextState = {
        ...state,
        phase: "live",
        roundEndsAt: now + state.pauseRemainingMs,
        pauseRemainingMs: 0,
        inputLocked: false,
        lastUpdatedAt: now,
      };

      await this.repository.saveState(nextState);

      return {
        round: state.round,
        resumedAt: now,
        endsAt: nextState.roundEndsAt,
      };
    });
  }

  async evaluateExpiredRound() {
    return this.lockService.withLock(
      "round-transition",
      async () => {
        const state = await this.repository.getState();
        const now = this.now();

        if (state.phase !== "live" || state.roundEndsAt > now) {
          return null;
        }

        return this.evaluateCurrentRound(state, now);
      },
      { returnNullOnBusy: true },
    );
  }

  async autoAdvanceIfReady() {
    if (!this.config.autoAdvanceRounds) {
      return null;
    }

    return this.lockService.withLock(
      "round-transition",
      async () => {
        const state = await this.repository.getState();
        const now = this.now();

        if (state.phase !== "results" || !state.autoAdvanceAt || state.autoAdvanceAt > now) {
          return null;
        }

        if (state.round >= state.totalRounds) {
          const finishedPayload = await this.finishGame(now);
          return {
            type: "game:finished",
            payload: finishedPayload,
          };
        }

        const payload = await this.startRound(state.round + 1, now);
        return {
          type: "round:started",
          payload,
        };
      },
      { returnNullOnBusy: true },
    );
  }

  async advanceRoundNow() {
    return this.lockService.withLock("game-admin", async () => {
      const now = this.now();
      const events = [];
      let state = await this.repository.getState();

      if (state.phase === "live" || state.phase === "paused") {
        const evaluation = await this.evaluateCurrentRound(state, now);
        events.push({ type: "round:evaluated", payload: evaluation });
        state = await this.repository.getState();
      }

      if (state.phase === "results" && state.round < state.totalRounds) {
        const started = await this.startRound(state.round + 1, now);
        events.push({ type: "round:started", payload: started });
      } else if (state.phase === "results" && state.round >= state.totalRounds) {
        const finished = await this.finishGame(now);
        events.push({ type: "game:finished", payload: finished });
      }

      return events;
    });
  }

  async evaluateCurrentRound(state, now) {
    if (state.phase === "results" || state.phase === "finished") {
      throw createError("ROUND_ALREADY_EVALUATED");
    }

    const signal = getSignalForRound(state.round);
    if (!signal) {
      throw createError("GAME_FINISHED");
    }

    const [teams, submissions] = await Promise.all([
      this.repository.listTeams(),
      this.repository.getSubmissions(state.round),
    ]);

    const updatedTeams = [];
    const results = [];

    for (const team of teams) {
      const submission = submissions[team.teamId] ?? null;
      const evaluation = evaluateDecision(signal, submission?.decision ?? null);
      const nextScore = team.score + evaluation.delta;

      updatedTeams.push({
        ...team,
        score: nextScore,
        lastDecision: submission?.decision ?? null,
        lastDelta: evaluation.delta,
        lastSubmittedRound: state.round,
        updatedAt: now,
      });

      results.push({
        teamId: team.teamId,
        name: team.name,
        decision: submission?.decision ?? null,
        submittedAt: submission?.submittedAt ?? null,
        verdict: evaluation.verdict,
        delta: evaluation.delta,
        score: nextScore,
      });
    }

    await this.repository.saveTeams(updatedTeams);

    const leaderboard = await this.buildLeaderboard(updatedTeams);
    const evaluationPayload = {
      round: state.round,
      evaluatedAt: now,
      signal: {
        ...toPublicSignal(signal),
        isAlpha: signal.isAlpha,
        correctDecision: signal.isAlpha ? "TRADE" : "IGNORE",
      },
      results,
      leaderboard,
    };

    await Promise.all([
      this.repository.saveLastEvaluation(evaluationPayload),
      this.repository.saveState({
        ...state,
        phase: "results",
        inputLocked: true,
        autoAdvanceAt: this.config.autoAdvanceRounds ? now + this.config.interRoundDelayMs : 0,
        lastUpdatedAt: now,
      }),
    ]);

    return evaluationPayload;
  }

  async startRound(round, now = this.now()) {
    const signal = getSignalForRound(round);

    if (!signal) {
      return this.finishGame(now);
    }

    await Promise.all([
      this.repository.clearRoundSubmissions(round),
      this.repository.saveState({
        phase: "live",
        round,
        totalRounds: SIGNALS.length,
        currentSignalId: signal.id,
        roundDurationMs: this.config.roundDurationMs,
        roundStartedAt: now,
        roundEndsAt: now + this.config.roundDurationMs,
        inputLocked: false,
        pauseRemainingMs: 0,
        autoAdvanceAt: 0,
        lastUpdatedAt: now,
      }),
    ]);

    return this.buildRoundStartedPayload(round, now);
  }

  async finishGame(now = this.now()) {
    const state = await this.repository.getState();
    await this.repository.saveState({
      ...state,
      phase: "finished",
      currentSignalId: "",
      inputLocked: true,
      autoAdvanceAt: 0,
      roundEndsAt: 0,
      pauseRemainingMs: 0,
      lastUpdatedAt: now,
    });

    return {
      finishedAt: now,
      leaderboard: await this.buildLeaderboard(await this.repository.listTeams()),
    };
  }

  buildRoundStartedPayload(round, now) {
    const signal = getSignalForRound(round);

    return {
      round,
      startedAt: now,
      endsAt: now + this.config.roundDurationMs,
      signal: toPublicSignal(signal),
    };
  }
}
