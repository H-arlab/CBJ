// Schema-only metadata for graph templates. Used for:
//   - LlmDock dispatch validation (`tpl in GRAPH_TPLS`)
//   - the template picker in GraphCell + recipe labels in canonicalRecipes
//   - axis labels / units / ticks rendered on the cell frame
// NO mock SVG paths / bands / boxes / bars / summaries — everything that
// shows up inside the plot area comes from /api/graphs/render with a
// real CSV dataset_id. If render fails or no dataset is bound, the cell
// shows an empty state, never a fake bezier curve.
export interface GraphTemplate {
  ey: string;
  title: string;
  yUnit: string;
  xUnit: string;
  yTicks: string[];
  xTicks: string[];
}

// IMPORTANT — H-Walker is a cable-driven assist robot. The columns
// L_ActForce_N / R_ActForce_N are the **loadcell force** at the cable,
// i.e. the assistive force the robot applies to each leg. They are
// NOT ground reaction force (GRF). Real GRF would come from a
// treadmill force-plate channel and is not part of these templates.
// Labels below say "Loadcell assist force" so users don't mis-cite
// these traces as GRF in publications.
export const GRAPH_TPLS: Record<string, GraphTemplate> = {
  // ── Cable / loadcell kinetics
  force: {
    ey: 'Loadcell · L vs R', title: 'Loadcell assist force (cable tension)',
    yUnit: 'Cable force (N)', xUnit: 'Gait cycle (%)',
    yTicks: ['60', '45', '30', '15', '0'], xTicks: ['0', '25', '50', '75', '100'],
  },
  force_avg: {
    ey: 'Loadcell · mean ± SD', title: 'Loadcell force, stride-averaged',
    yUnit: 'Cable force (N)', xUnit: 'Gait cycle (%)',
    yTicks: ['60', '45', '30', '15', '0'], xTicks: ['0', '25', '50', '75', '100'],
  },
  force_lr_subplot: {
    ey: 'Loadcell · L / R subplots', title: 'Loadcell force per leg (GCP-normalized)',
    yUnit: 'Cable force (N)', xUnit: 'Gait cycle (%)',
    yTicks: ['60', '45', '30', '15', '0'], xTicks: ['0', '25', '50', '75', '100'],
  },
  asymmetry: {
    ey: 'Asymmetry · per stride', title: 'Asymmetry index across strides',
    yUnit: 'Asymmetry (%)', xUnit: 'Stride #',
    yTicks: ['10', '7.5', '5', '2.5', '0'], xTicks: ['1', '4', '7', '10', '14'],
  },
  // (Removed: `cop` and `cv_bar`. Both have no real-data renderer
  // yet — `cop` will return when the Motion-source force-plate
  // pipeline lands, `cv_bar` when multi-trial structure ships.)
  trials: {
    ey: 'Trials overlay', title: 'Trial-by-trial overlay',
    yUnit: 'Normalized force', xUnit: 'Gait cycle (%)',
    yTicks: ['1.0', '0.75', '0.5', '0.25', '0'], xTicks: ['0', '25', '50', '75', '100'],
  },

  // ── Kinematics (joint-specific labels enforced in the picker UI)
  imu_avg: {
    ey: 'Kinematics · mean ± SD', title: 'Joint angle over gait cycle (per channel)',
    yUnit: 'Joint angle (°)', xUnit: 'Gait cycle (%)',
    yTicks: ['+20', '+10', '0', '−10', '−20'], xTicks: ['0', '25', '50', '75', '100'],
  },
  cyclogram: {
    ey: 'Phase portrait', title: 'Joint-vs-joint cyclogram',
    yUnit: 'Distal joint (°)', xUnit: 'Proximal joint (°)',
    yTicks: ['+30', '+15', '0', '−15', '−30'], xTicks: ['−20', '−10', '0', '+10', '+20'],
  },
  rom_bar: {
    ey: 'Kinematics · ROM by joint × plane',
    title: 'Range of motion (joint × plane)',
    yUnit: 'ROM (°)', xUnit: 'joint · plane',
    yTicks: ['60', '45', '30', '15', '0'], xTicks: ['L sag', 'L fro', 'R sag', 'R fro'],
  },

  // ── Temporal / phase
  stride_time_trend: {
    ey: 'Temporal · stride time over time', title: 'Stride time across strides',
    yUnit: 'Stride T (s)', xUnit: 'Stride #',
    yTicks: ['1.20', '1.10', '1.00', '0.90', '0.80'], xTicks: ['1', '5', '10', '15', '20'],
  },
  stance_swing_bar: {
    ey: 'Temporal phases', title: 'Stance / swing percentages',
    yUnit: '% gait cycle', xUnit: '',
    yTicks: ['100', '75', '50', '25', '0'], xTicks: ['L stance', 'L swing', 'R stance', 'R swing'],
  },

  // ── Symmetry
  symmetry_radar: {
    ey: 'Symmetry', title: 'Symmetry summary (0 = perfect)',
    yUnit: 'Asymmetry (%)', xUnit: '',
    yTicks: ['10', '7.5', '5', '2.5', '0'], xTicks: ['T', 'L', 'St', 'F', 'Pk'],
  },
};
