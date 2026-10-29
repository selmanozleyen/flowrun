# Shared overrides for the new-only flow (sourced by the job scripts).
N=/lustre/groups/ml01/datasets/$USER/citeseq_panel/prepared_newonly
NEWONLY=(+experiment=citeseq "paths.target_adata=[$N/old_panel.zarr,$N/new_panel.zarr]"
  datamanager.target_sample_rep=new_only datamanager.source_sample_rep=shared_recal
  datamanager.lin_from_state=false datamanager.quad_from_state=false
  datamanager.rep_lin=lin datamanager.source_rep_quad=quad_old datamanager.target_rep_quad=quad_new
  coupling.epsilon=0.1 coupling.alpha=0.25
  module.state_encoder_output_dim=64 module.time_encoder_output_dim=64
  'module.state_encoder_mlp_kwargs.hidden_dims=[128]' 'module.time_encoder_mlp_kwargs.hidden_dims=[128]'
  'module.vf_decoder_mlp_kwargs.hidden_dims=[256,256]'
  module.source_encoder_output_dim=64 'module.source_encoder_mlp_kwargs.hidden_dims=[128]')
