/* API confirmed against archived 4.3.2 and official 4.3.5 sources.
 * Copy to MO/MSSM/omega_driver.c, then make main=omega_driver.c.
 * External spectrum only: no SuSpect/SPheno/SoftSUSY invocation.
 */
#include "../include/micromegas.h"
#include "../include/micromegas_aux.h"
#include "lib/pmodel.h"
#include "mo_build_version.h"
#include <errno.h>
#include <limits.h>

static const char *result_path;

static int result_error(const char *status, const char *reason, int exit_code)
{
    FILE *out = fopen(result_path, "w");
    fprintf(stderr, "%s: %s\n", status, reason);
    if (out) {
        /* Both strings here are fixed literals, not external input. */
        fprintf(out, "{\"status\":\"%s\",\"reason\":\"%s\","
                     "\"source_version\":\"" MO_SOURCE_VERSION "\","
                     "\"paper_version_match\":" MO_PAPER_VERSION_MATCH "}\n",
                     status, reason);
        fclose(out);
    }
    return exit_code;
}

static int required(char *block, int n, int i, int j)
{
    int exists = n == 0 ? slhaValExists(block, 0) :
                 n == 1 ? slhaValExists(block, 1, i) :
                          slhaValExists(block, 2, i, j);
    double value;
    if (!exists) {
        fprintf(stderr, "Missing required SLHA entry %s %d %d\n", block, i, j);
        return 0;
    }
    value = n == 0 ? slhaVal(block, 91., 0) :
            n == 1 ? slhaVal(block, 91., 1, i) :
                     slhaVal(block, 91., 2, i, j);
    return isfinite(value);
}

static int preflight(char *filename)
{
    int i, j, k;
    const int mass[] = {25,35,36,37,1000021,1000022,1000023,1000025,1000035,
        1000024,1000037,1000011,2000011,1000013,2000013,1000015,2000015,
        1000012,1000014,1000016,1000001,2000001,1000002,2000002,
        1000003,2000003,1000004,2000004,1000005,2000005,1000006,2000006};
    const int soft[] = {1,2,3,31,32,33,34,35,36,41,42,43,44,45,46,47,48,49};
    const int sm[] = {3,5,6,7};
    char *mix[] = {"NMIX","UMIX","VMIX","STOPMIX","SBOTMIX","STAUMIX"};
    char *tri[] = {"AU","AD","AE"};
    FError = 0;
    if (slhaRead(filename, 0)) return 0;
    for (i=0; i<(int)(sizeof(mass)/sizeof(mass[0])); ++i)
        if (!required("MASS",1,mass[i],0)) return 0;
    for (i=0; i<(int)(sizeof(soft)/sizeof(soft[0])); ++i)
        if (!required("MSOFT",1,soft[i],0)) return 0;
    for (i=0; i<4; ++i) if (!required("SMINPUTS",1,sm[i],0)) return 0;
    if (!required("ALPHA",0,0,0) || !required("HMIX",1,1,0) ||
        !required("HMIX",1,2,0)) return 0;
    for (k=0; k<6; ++k) {
        int dimension = k == 0 ? 4 : 2;
        for (i=1; i<=dimension; ++i) for (j=1; j<=dimension; ++j)
            if (!required(mix[k],2,i,j)) return 0;
    }
    for (k=0; k<3; ++k) if (!required(tri[k],2,3,3)) return 0;
    /* 4.3.5 FillVal has an AE22 fallback spelling "Al"; require AE22
       explicitly so missing data cannot silently select that fallback. */
    if (!required("AE",2,2,2)) return 0;
    return !FError;
}

static int parse_int(char *text, int *value)
{
    char *end;
    long v;
    errno=0;
    v=strtol(text,&end,10);
    if (errno || !*text || *end || v< -1 || v>2) return 0;
    *value=(int)v;
    return 1;
}

int main(int argc, char **argv)
{
    int err, spin2, charge3, color, fast=1, vw=0, vz=0;
    char cdm[64] = {0};
    char *end;
    double beps=1.E-5, xf=25., omega;
    FILE *out;
    char workspace[PATH_MAX];
    const char *local_work;
    if (argc!=3 && argc!=7) {
        fprintf(stderr,"Usage: omega_driver SPECTRUM_SLHA RESULT_JSON [FAST BEPS VW VZ]\n");
        return 64;
    }
    result_path=argv[2];
    if (argc==7) {
        errno=0;
        beps=strtod(argv[4],&end);
        if (errno || !*argv[4] || *end || !isfinite(beps) || beps<=0 || beps>1 ||
            !parse_int(argv[3],&fast) || fast>1 ||
            (fast<0 && !MO_HAS_ROBUST_FAST) ||
            !parse_int(argv[5],&vw) || vw<0 ||
            !parse_int(argv[6],&vz) || vz<0)
            return result_error("technical_failure","invalid_numeric_settings",64);
    }
    /* Dynamic process generation uses WORK. Override only when a caller
       supplies a complete, isolated copy of the built MSSM/work directory. */
    local_work=getenv("MO_WORK_DIR");
    if (local_work) {
        if (!realpath(local_work,workspace))
            return result_error("technical_failure","invalid_mo_work_dir",65);
        WORK=workspace;
    }
    if (!preflight(argv[1]))
        return result_error("technical_failure","invalid_or_incomplete_slha",66);
    FError=0;
    err=lesHinput(argv[1]);
    if (err || FError)
        return result_error("technical_failure","lesHinput_failed",67);
    slhaWarnings(stdout);
    ForceUG=0;
    err=sortOddParticles(cdm);
    if (err || FError)
        return result_error("technical_failure","sortOddParticles_failed",68);
    qNumbers(cdm,&spin2,&charge3,&color);
    if (charge3 || color!=1)
        return result_error("unphysical","lsp_is_charged_or_colored",0);
    if (strcmp(cdm,"~o1"))
        return result_error("unphysical","lsp_is_not_lightest_neutralino",0);
    if (!isfinite(Mcdm) || Mcdm<=0)
        return result_error("technical_failure","invalid_dark_matter_mass",69);
    VWdecay=vw;
    VZdecay=vz;
    cleanDecayTable();
    FError=0;
    omega=darkOmega(&xf,fast,beps);
    if (FError || !isfinite(omega) || omega<=0 || !isfinite(xf) || xf<=0) {
        fprintf(stderr,"darkOmega diagnostics: omega=%.17g xf=%.17g FError=%d\n",
                omega,xf,FError);
        return result_error("technical_failure","darkOmega_failed",70);
    }
    out=fopen(result_path,"w");
    if (!out) { perror("result JSON"); return 74; }
    fprintf(out,"{\n  \"status\":\"ok\", \"source_version\":\"" MO_SOURCE_VERSION "\", "
                "\"paper_version_match\":" MO_PAPER_VERSION_MATCH ",\n"
                "  \"omega_h2\":%.17g, \"xf\":%.17g,\n"
                "  \"cdm_name\":\"~o1\", \"mcdm\":%.17g,\n"
                "  \"settings\":{\"fast\":%d,\"beps\":%.17g,\"vw\":%d,\"vz\":%d},\n"
                "  \"imported\":{\"Mh\":%.17g,\"MH\":%.17g,\"MH3\":%.17g,"
                "\"MHc\":%.17g,\"MNE1\":%.17g,\"alpha\":%.17g,"
                "\"Zn11\":%.17g,\"Zn12\":%.17g,\"Zn13\":%.17g,\"Zn14\":%.17g,"
                "\"mu\":%.17g,\"tan_beta\":%.17g}\n}\n",
            omega,xf,Mcdm,fast,beps,vw,vz,findValW("Mh"),findValW("MH"),
            findValW("MH3"),findValW("MHc"),findValW("MNE1"),findValW("alpha"),
            findValW("Zn11"),findValW("Zn12"),findValW("Zn13"),findValW("Zn14"),
            findValW("mu"),findValW("tB"));
    if (fclose(out)) return 74;
    return 0;
}
