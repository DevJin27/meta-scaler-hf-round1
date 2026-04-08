import { z } from "zod";

export const teamJoinSchema = z.object({
  teamId: z.string().trim().min(1).max(64),
  name: z.string().trim().min(1).max(120),
});

export const decisionSchema = z.object({
  decision: z.enum(["TRADE", "IGNORE"]),
  round: z.coerce.number().int().positive().optional(),
});

export const adminAuthSchema = z.object({
  secret: z.string().trim().min(1),
});

export function normalizeTeamJoin(payload) {
  return teamJoinSchema.parse(payload);
}

export function normalizeDecision(payload) {
  const parsed = decisionSchema.parse(payload);
  return {
    ...parsed,
    decision: parsed.decision.toUpperCase(),
  };
}

export function normalizeAdminAuth(payload) {
  return adminAuthSchema.parse(payload);
}
