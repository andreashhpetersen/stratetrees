int shield(double x, double t) {
   if (t <= 0.64) {
    if (t <= 0.45999999999999996) {
      if (t <= 0.37) {
        if (t <= 0.28) {
          if (t <= 0.12) {
            return 2;
          } else {
            if (x <= 0.035) {
              return 1;
            } else {
              if (t <= 0.21) {
                return 2;
              } else {
                if (x <= 0.16) {
                  return 1;
                } else {
                  return 2;
                }
              }
            }
          }
        } else {
          if (x <= 0.16) {
            if (x <= 0.09) {
              return 0;
            } else {
              return 1;
            }
          } else {
            if (t <= 0.3) {
              return 2;
            } else {
              if (x <= 0.29000000000000004) {
                return 1;
              } else {
                return 2;
              }
            }
          }
        }
      } else {
        if (x <= 0.29000000000000004) {
          if (x <= 0.09) {
            return 0;
          } else {
            if (x <= 0.22) {
              return 0;
            } else {
              return 1;
            }
          }
        } else {
          if (t <= 0.39) {
            return 2;
          } else {
            if (x <= 0.42000000000000004) {
              return 1;
            } else {
              return 2;
            }
          }
        }
      }
    } else {
      if (x <= 0.35) {
        return 0;
      } else {
        if (x <= 0.5499999999999999) {
          if (t <= 0.5499999999999999) {
            if (x <= 0.42000000000000004) {
              return 1;
            } else {
              if (t <= 0.48) {
                return 2;
              } else {
                return 1;
              }
            }
          } else {
            if (x <= 0.48) {
              return 0;
            } else {
              return 1;
            }
          }
        } else {
          if (t <= 0.57) {
            return 2;
          } else {
            if (x <= 0.6799999999999999) {
              return 1;
            } else {
              return 2;
            }
          }
        }
      }
    }
  } else {
    if (x <= 0.81) {
      if (x <= 0.61) {
        return 0;
      } else {
        if (t <= 0.73) {
          if (x <= 0.6799999999999999) {
            return 1;
          } else {
            if (t <= 0.66) {
              return 2;
            } else {
              return 1;
            }
          }
        } else {
          if (x <= 0.74) {
            return 0;
          } else {
            if (t <= 0.8200000000000001) {
              return 1;
            } else {
              return 0;
            }
          }
        }
      }
    } else {
      if (x <= 0.9999999999999999) {
        if (t <= 0.75) {
          return 2;
        } else {
          if (x <= 0.94) {
            if (t <= 0.8200000000000001) {
              return 1;
            } else {
              if (x <= 0.87) {
                return 0;
              } else {
                if (t <= 0.9099999999999999) {
                  return 1;
                } else {
                  return 0;
                }
              }
            }
          } else {
            if (t <= 0.84) {
              return 2;
            } else {
              if (t <= 0.9099999999999999) {
                return 1;
              } else {
                return 0;
              }
            }
          }
        }
      } else {
        if (t <= 0.75) {
          return 2;
        } else {
          if (t <= 0.84) {
            return 2;
          } else {
            if (t <= 0.9999999999999999) {
              return 2;
            } else {
              return 0;
            }
          }
        }
      }
    }
  }

}