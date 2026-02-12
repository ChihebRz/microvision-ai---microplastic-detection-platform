
export interface MicroplasticAnnotation {
  id: string;
  class: string;
  bbox: [number, number, number, number]; // [x, y, w, h] normalized
  confidence?: number;
}

export interface DatasetItem {
  id: string;
  imageUrl: string;
  annotations: MicroplasticAnnotation[];
  metadata: {
    source: string;
    location: string;
    timestamp: string;
  };
}

export interface TrainingMetric {
  epoch: number;
  loss: number;
  valLoss: number;
  mAP: number;
  precision: number;
  recall: number;
}

export enum Page {
  Dashboard = 'dashboard',
  Dataset = 'dataset',
  Training = 'training',
  Prediction = 'prediction',
  Insights = 'insights'
}
