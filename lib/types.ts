export type StageId = "stage-1" | "stage-2" | "stage-3";

export interface ProductStage {
  id: StageId;
  name: string;
  status: "now" | "next" | "later";
  headline: string;
  goals: string[];
}
