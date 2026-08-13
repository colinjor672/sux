shader_type canvas_item;

render_mode unshaded, blend_premul_alpha;

varying vec2 ribbon_uv;


uniform bool near_at_start = true;

uniform bool draw_glow = true;
uniform bool draw_center_light = true;
uniform bool draw_flow_light = true;
uniform bool draw_edge_lines = true;


uniform float near_width = 130.0;
uniform float far_width = 60.2;

uniform float glow_extra_width = 60.0;


uniform float center_light_near_width = 60.2;
uniform float center_light_far_width = 22.4;


uniform vec4 near_color : source_color =
	vec4(0.05, 0.85, 1.0, 0.60);

uniform vec4 far_color : source_color =
	vec4(0.05, 0.55, 1.0, 0.10);


uniform vec4 edge_near_color : source_color =
	vec4(0.0, 0.95, 1.0, 0.35);

uniform vec4 edge_far_color : source_color =
	vec4(0.0, 0.45, 1.0, 0.00);

uniform float edge_line_width_near = 5.0;
uniform float edge_line_width_far = 2.5;
uniform float edge_line_inset = 3.0;

uniform vec4 edge_line_near_color : source_color =
	vec4(0.0, 0.85, 1.0, 0.95);

uniform vec4 edge_line_far_color : source_color =
	vec4(0.0, 0.55, 1.0, 0.25);


uniform vec4 glow_near_color : source_color =
	vec4(0.0, 0.75, 1.0, 0.45);

uniform vec4 glow_far_color : source_color =
	vec4(0.0, 0.35, 1.0, 0.08);


uniform vec4 center_light_near_color : source_color =
	vec4(0.75, 1.0, 1.0, 0.45);

uniform vec4 center_light_far_color : source_color =
	vec4(0.75, 1.0, 1.0, 0.08);


uniform float flow_speed = 150.0;
uniform float flow_segment_length = 120.0;
uniform float flow_segment_gap = 170.0;

uniform bool reverse_flow_direction = false;


uniform vec4 flow_near_color : source_color =
	vec4(0.9, 1.0, 1.0, 0.85);

uniform vec4 flow_far_color : source_color =
	vec4(0.6, 0.9, 1.0, 0.08);


uniform float path_length = 1.0;


void vertex() {
	ribbon_uv = UV;
}


float smooth01(float x) {
	x = clamp(x, 0.0, 1.0);

	return x * x * (
		3.0 - 2.0 * x
	);
}


float get_perspective_t(float path_t) {
	path_t = clamp(path_t, 0.0, 1.0);

	if (near_at_start) {
		return path_t;
	}

	return 1.0 - path_t;
}


float ribbon_alpha(float normalized_side) {
	float x = clamp(
		normalized_side,
		0.0,
		1.0
	);

	if (x <= 0.48) {
		float t = smoothstep(
			0.0,
			0.48,
			x
		);

		return mix(
			1.0,
			0.86,
			t
		);
	}

	if (x <= 0.82) {
		float t = smoothstep(
			0.48,
			0.82,
			x
		);

		return mix(
			0.86,
			0.28,
			t
		);
	}

	float t = smoothstep(
		0.82,
		1.0,
		x
	);

	return mix(
		0.28,
		0.0,
		t
	);
}


vec4 make_ribbon_layer(
	float cross_abs,
	float width_ratio,
	float perspective_t,

	vec4 core_near,
	vec4 core_far,

	vec4 edge_near,
	vec4 edge_far
) {
	width_ratio = max(
		width_ratio,
		0.0001
	);

	float normalized_side = (
		cross_abs
		/ width_ratio
	);

	if (normalized_side >= 1.0) {
		return vec4(0.0);
	}

	float alpha_profile = ribbon_alpha(
		normalized_side
	);

	vec4 core_color = mix(
		core_near,
		core_far,
		perspective_t
	);

	vec4 edge_color = mix(
		edge_near,
		edge_far,
		perspective_t
	);

	float edge_mix = smoothstep(
		0.15,
		1.0,
		normalized_side
	);

	vec4 color = mix(
		core_color,
		edge_color,
		edge_mix
	);

	color.a *= alpha_profile;

	return color;
}


float get_flow_mask(float distance_on_path) {
	float cycle = max(
		1.0,
		flow_segment_length
		+ flow_segment_gap
	);

	float offset = mod(
		TIME * flow_speed,
		cycle
	);

	if (reverse_flow_direction) {
		offset = cycle - offset;
	}

	float local_distance = mod(
		distance_on_path
		- offset
		+ cycle,
		cycle
	);

	if (local_distance >= flow_segment_length) {
		return 0.0;
	}

	float segment_t = clamp(
		local_distance
		/ max(flow_segment_length, 0.001),
		0.0,
		1.0
	);

	return sin(
		segment_t
		* 3.14159265359
	);
}


void fragment() {
	/*
		手工 Mesh：
		ribbon_uv.x = 横向
		ribbon_uv.y = 路径方向
	*/

	float across_t = clamp(
		ribbon_uv.x,
		0.0,
		1.0
	);

	float along_t = clamp(
		ribbon_uv.y,
		0.0,
		1.0
	);

	float perspective_t = get_perspective_t(
		along_t
	);

	float width_t = smooth01(
		perspective_t
	);


	float geometry_width = mix(
		near_width + glow_extra_width,
		far_width + glow_extra_width,
		width_t
	);

	float body_width = mix(
		near_width,
		far_width,
		width_t
	);

	float center_width = mix(
		center_light_near_width,
		center_light_far_width,
		width_t
	);


	float body_ratio = clamp(
		body_width
		/ max(geometry_width, 0.001),
		0.0,
		1.0
	);

	float center_ratio = clamp(
		center_width
		/ max(geometry_width, 0.001),
		0.0,
		1.0
	);


	float cross_abs = abs(
		across_t * 2.0 - 1.0
	);


	vec4 glow = vec4(0.0);
	vec4 body = vec4(0.0);
	vec4 center = vec4(0.0);
	vec4 flow = vec4(0.0);
	vec4 border = vec4(0.0);


	if (draw_glow) {
		vec4 glow_edge_near = glow_near_color;
		vec4 glow_edge_far = glow_far_color;

		glow_edge_near.a *= 0.55;

		glow = make_ribbon_layer(
			cross_abs,
			1.0,
			perspective_t,

			glow_near_color,
			glow_far_color,

			glow_edge_near,
			glow_edge_far
		);
	}


	body = make_ribbon_layer(
		cross_abs,
		body_ratio,
		perspective_t,

		near_color,
		far_color,

		edge_near_color,
		edge_far_color
	);


	if (draw_center_light) {
		center = make_ribbon_layer(
			cross_abs,
			center_ratio,
			perspective_t,

			center_light_near_color,
			center_light_far_color,

			center_light_near_color,
			center_light_far_color
		);
	}


	if (draw_flow_light) {
		float flow_ratio = clamp(
			center_ratio * 1.35,
			0.0,
			1.0
		);

		float distance_on_path = (
			along_t
			* max(path_length, 1.0)
		);

		float flow_mask = get_flow_mask(
			distance_on_path
		);

		flow = make_ribbon_layer(
			cross_abs,
			flow_ratio,
			perspective_t,

			flow_near_color,
			flow_far_color,

			flow_near_color,
			flow_far_color
		);

		flow.a *= flow_mask;
	}


	if (draw_edge_lines) {
		float half_geometry_width = geometry_width * 0.5;
		float distance_from_mesh_edge = (
			(1.0 - cross_abs) * half_geometry_width
		);
		float line_width = mix(
			edge_line_width_near,
			edge_line_width_far,
			width_t
		);
		float line_distance = abs(
			distance_from_mesh_edge - edge_line_inset
		);
		float line_alpha = 1.0 - smoothstep(
			max(line_width * 0.5 - 1.0, 0.0),
			line_width * 0.5 + 1.0,
			line_distance
		);

		border = mix(
			edge_line_near_color,
			edge_line_far_color,
			width_t
		);
		border.a *= line_alpha;
	}


	/*
		恢复最初版本的视觉合成：
		每层 RGB 乘自身 alpha 后相加。
	*/

	vec3 rgb = vec3(0.0);

	rgb += glow.rgb * glow.a;
	rgb += body.rgb * body.a;
	rgb += center.rgb * center.a;
	rgb += flow.rgb * flow.a * 1.25;
	rgb += border.rgb * border.a;


	float alpha = clamp(
		glow.a * 0.55
		+ body.a
		+ center.a * 0.85
		+ flow.a
		+ border.a,
		0.0,
		1.0
	);


	if (alpha <= 0.0001) {
		discard;
	}


	COLOR = vec4(
		rgb,
		alpha
	);
}
