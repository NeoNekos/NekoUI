Texture2D<float> glyph_atlas : register(t0);
SamplerState glyph_sampler : register(s0);

struct FragmentInput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
};

float4 fs_main(FragmentInput input) : SV_Target0 {
    float mask = glyph_atlas.Sample(glyph_sampler, input.uv);
    return float4(input.color.rgb, input.color.a * mask);
}
