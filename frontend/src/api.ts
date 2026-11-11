
import axios from 'axios';

const API_SERVER = import.meta.env.VITE_API_SERVER || import.meta.env.VITE_API_SERVER || "https://tools.kenarnold.org/api";

export interface Highlight {
  start: number;
  end: number;
  token: string;
  token_loss: number;
  most_likely_token: string;
  topk_tokens: string[];
  loss_ratio?: number;
}

export interface NextToken {
  token: string;
  logprob: number;
}

export interface LogprobEntry {
  token: string | null;
  logprobs: Record<string, number>;
}

export const getPredsApi = async (prompt: string, originalDoc: string, docInProgress: string, k: number = 5) => {
  const response = await axios.get(`${API_SERVER}/next_token`, {
    params: {
      prompt,
      original_doc: originalDoc,
      doc_in_progress: docInProgress,
      k
    }
  });
  return response.data.next_tokens; // returns list of strings
};

export const getHighlights = async (prompt: string, doc: string, updatedDoc: string) => {
  const response = await axios.get(`${API_SERVER}/highlights`, {
    params: {
      prompt,
      doc,
      updated_doc: updatedDoc
    }
  });
  return response.data.highlights as Highlight[];
};

export const continueMessages = async (messages: any[], n_branch_tokens: number = 5, n_future_tokens: number = 2) => {
  const response = await axios.post(`${API_SERVER}/continue_messages`, {
    messages,
    n_branch_tokens,
    n_future_tokens
  });
  return response.data.continuations;
};

export const getLogprobs = async (messages: any[], n_branch_tokens: number = 5, n_future_tokens: number = 2) => {
  const response = await axios.post(`${API_SERVER}/logprobs`, {
    messages,
    n_branch_tokens,
    n_future_tokens
  });
  return response.data.logprobs as LogprobEntry[];
};
