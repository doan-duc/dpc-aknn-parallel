import os

html = """<!doctype html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <title>Báo cáo Tối ưu hóa DPC-AKNN</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">

    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/reset.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/reveal.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/theme/black.min.css" id="theme">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@500;700;900&family=Inter:wght@300;400;500;600&family=Fira+Code:wght@400;600&display=swap" rel="stylesheet">

    <style>
        :root {
            --neon-blue: #00f3ff;
            --neon-purple: #b537f2;
            --neon-red: #ff3366;
            --neon-green: #00ff66;
            --neon-yellow: #ffde00;
            --bg-dark: #050508;
            --glass-bg: rgba(15, 15, 20, 0.85);
            --glass-border: rgba(255, 255, 255, 0.15);
        }

        body {
            background: var(--bg-dark);
            font-family: 'Inter', sans-serif;
            color: #ffffff;
            text-align: left;
        }

        .reveal { font-size: 22px; }
        
        .reveal h1, .reveal h2, .reveal h3, .reveal h4 {
            font-family: 'Montserrat', sans-serif;
            text-align: left;
            line-height: 1.3;
            color: #ffffff;
        }

        .reveal h1 { font-size: 2.2em; font-weight: 900; margin-bottom: 20px;}
        .reveal h2 { font-size: 1.5em; font-weight: 700; margin-bottom: 15px; border-bottom: 3px solid #333; padding-bottom: 10px;}
        
        .reveal code {
            font-family: 'Fira Code', monospace;
            background: rgba(255,255,255,0.1);
            color: var(--neon-yellow);
            padding: 2px 6px;
            border-radius: 4px;
        }

        .glass-panel {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.8);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .slide-layout {
            display: grid;
            grid-template-columns: 45% 55%;
            gap: 25px;
            height: 65vh;
            align-items: stretch;
        }
        
        .viz-box {
            background: #000;
            border-radius: 12px;
            position: relative;
            overflow: hidden;
            border: 2px solid #333;
            box-shadow: inset 0 0 40px rgba(0,0,0,1);
            width: 100%;
            height: 100%;
        }

        canvas.scene-canvas {
            width: 100%;
            height: 100%;
            display: block;
        }

        .v-tag {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 900;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 0 15px rgba(0,0,0,0.8);
        }
        .v-goc { background: #222; color: #aaa; border: 2px solid #555; }
        .v-1 { background: rgba(0, 243, 255, 0.1); color: var(--neon-blue); border: 2px solid var(--neon-blue);}
        .v-2 { background: rgba(255, 222, 0, 0.1); color: var(--neon-yellow); border: 2px solid var(--neon-yellow);}
        .v-final { background: rgba(0, 255, 102, 0.1); color: var(--neon-green); border: 2px solid var(--neon-green);}

        ul.custom-list { list-style: none; padding: 0; margin: 0; text-align: left; }
        ul.custom-list li { margin-bottom: 18px; padding-left: 40px; position: relative; font-size: 0.95em; line-height: 1.6; font-weight: 500;}
        ul.custom-list li i { position: absolute; left: 0; top: 4px; font-size: 1.2em; color: var(--neon-blue);}

        .text-blue { color: var(--neon-blue); text-shadow: 0 0 15px rgba(0,243,255,0.5); }
        .text-green { color: var(--neon-green); text-shadow: 0 0 15px rgba(0,255,102,0.5); }
        .text-red { color: var(--neon-red); text-shadow: 0 0 15px rgba(255,51,102,0.5); }
        .text-yellow { color: var(--neon-yellow); text-shadow: 0 0 15px rgba(255,222,0,0.5); }
        .text-purple { color: var(--neon-purple); text-shadow: 0 0 15px rgba(181,55,242,0.5); }
    </style>
</head>
<body>
    <div class="reveal">
        <div class="slides">

            <!-- SLIDE 1 -->
            <section>
                <div style="margin-bottom: 30px; text-align: center;">
                    <i class="fa-solid fa-microchip text-blue" style="font-size: 5em;"></i>
                </div>
                <h1 style="text-align: center;" class="text-blue">DPC-AKNN CORE</h1>
                <h3 style="text-align: center;">Báo cáo Tối ưu hóa Đa luồng C++</h3>
                <p style="text-align: center; color: #888;">Trực quan hóa Thuật toán & Khắc phục Cổ chai Phần cứng</p>
            </section>

            <!-- SLIDE 2: CANVAS SIMULATION -->
            <section>
                <h2 class="text-blue">Mô phỏng Toàn cảnh Thuật toán</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-play"></i> <b>Bước 1:</b> kNN (Láng giềng).</li>
                            <li><i class="fa-solid fa-play"></i> <b>Bước 2-3:</b> Bán kính d_c & Mật độ cục bộ.</li>
                            <li><i class="fa-solid fa-play"></i> <b>Bước 4:</b> Tìm Tâm cụm.</li>
                            <li><i class="fa-solid fa-play"></i> <b>Bước 5:</b> Xây cụm ban đầu (Mở rộng cụm).</li>
                            <li><i class="fa-solid fa-play"></i> <b>Bước 6-8:</b> Ma trận A, Bầu chọn ranh giới, Ngoại lai.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="master-sim" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <!-- SLIDE 3: GOALS -->
            <section>
                <h2 class="text-green">Mục tiêu Cốt lõi</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-memory text-red"></i> <b>Bóp nghẹt RAM:</b> Xóa bỏ Ma trận O(N²) phá hủy bộ nhớ.</li>
                            <li><i class="fa-solid fa-microchip text-blue"></i> <b>Khai phóng CPU:</b> Chống False Sharing và Lock Contention.</li>
                            <li><i class="fa-solid fa-bullseye text-green"></i> <b>Bảo toàn ARI:</b> Giữ nguyên kết quả 100% so với tuần tự.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="goals" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <!-- ================= BƯỚC 1: kNN ================= -->
            
            <section>
                <h1 class="text-blue">BƯỚC 1: TÍNH kNN</h1>
                <p>Mắt xích yếu nhất, chiếm 90% thời gian thực thi của cả chương trình.</p>
                <div class="viz-box" style="height:400px; margin-top:20px;">
                    <canvas class="scene-canvas" data-scene="knn-intro" width="1000" height="400"></canvas>
                </div>
            </section>

            <section>
                <h2 class="text-red">Bước 1 - Vấn đề: Tràn RAM (Version Gốc)</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-goc">Version Gốc</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-database"></i> Khai báo Ma trận <code>D[N×N]</code>.</li>
                            <li><i class="fa-solid fa-triangle-exclamation"></i> 70.000 điểm tốn tới <b>39.2 GB RAM</b> vật lý.</li>
                            <li>Hệ điều hành tràn RAM, treo máy và Swap ổ cứng.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="ram-oom" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <section>
                <h2 class="text-purple">Bước 1 - Phân tích: Memory Wall</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-goc">Version Gốc</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-microchip"></i> CPU tính toán cực nhanh (Chỉ 0.5ns/phép tính).</li>
                            <li><i class="fa-solid fa-memory"></i> Gọi dữ liệu từ RAM mất tới 100ns (Chậm gấp 200 lần).</li>
                            <li>Việc <b>ĐỌC</b> Ma trận D khổng lồ khiến CPU đói dữ liệu.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="mem-wall" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <section>
                <h2 class="text-blue">Bước 1 - Giải pháp: On-The-Fly (Version 1)</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-1">Version 1</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-fire"></i> Tính đến đâu, vứt đến đó! Không bao giờ lưu mảng N×N.</li>
                            <li><i class="fa-solid fa-filter"></i> Chỉ giữ lại K láng giềng nhỏ nhất trên khay nhớ tạm.</li>
                            <li>RAM giảm từ 39 GB xuống <b>vài MB</b>.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="on-the-fly" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <section>
                <h2 class="text-yellow">Bước 1 - Cổ chai: Lock Contention (Version 2)</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-2">Version 2</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-lock"></i> Gọi hàm <code>malloc()</code> TRONG vòng lặp song song.</li>
                            <li><i class="fa-solid fa-ban"></i> Luồng phải xếp hàng chờ OS cấp RAM (Mutex Lock).</li>
                            <li>Hệ thống 16 nhân bị nghẽn, chạy chậm hơn 1 nhân!</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="lock-contention" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <section>
                <h2 class="text-green">Bước 1 - Giải pháp: Pre-Allocated Buffers (Final)</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-final">Final Version</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-road"></i> Cấp phát tĩnh mảng <code>bufs[T]</code> TRƯỚC vùng song song.</li>
                            <li><i class="fa-solid fa-user-ninja"></i> Luồng lấy buffer bằng <code>omp_get_thread_num()</code>.</li>
                            <li>Băng thông CPU mở khóa 100%. Không Lock.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="pre-alloc" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <section>
                <h2 class="text-red">Bước 1 - Vấn đề: Thuật toán Qsort O(N log N)</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-goc">Version Gốc / 1 / 2</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-sort"></i> Dùng <code>qsort()</code> cho mảng độ dài N (70,000) chỉ để lấy K.</li>
                            <li><i class="fa-solid fa-clock"></i> Thuật toán O(N log N) lặp lại liên tục N lần gây lãng phí xung nhịp CPU.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="qsort-nlogn" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <section>
                <h2 class="text-green">Bước 1 - Giải pháp: Max-Heap (Final)</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-final">Final Version</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-tree"></i> Dùng <b>Max-Heap</b> có kích thước đúng K.</li>
                            <li><i class="fa-solid fa-arrows-down-to-line"></i> Điểm mới < Root ➔ Hất Root, nạp điểm mới O(log K).</li>
                            <li>Độ phức tạp giảm thê thảm từ O(N log N) xuống O(N log K).</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="max-heap" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <section>
                <h2 class="text-green">Bước 1 - Hack Toán Học: Early Exit (Final)</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-final">Final Version</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-square-root-variable"></i> Loại bỏ hàm <code>sqrt()</code>, so sánh thẳng bằng khoảng cách bình phương.</li>
                            <li><i class="fa-solid fa-scissors"></i> <b>Early Exit:</b> Nếu cộng dồn x, y đã lớn hơn Max-Heap ➔ Break ngay lập tức.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="early-exit" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <!-- ================= BƯỚC 2: d_c ================= -->

            <section>
                <h1 class="text-blue">BƯỚC 2: TÍNH BÁN KÍNH d_c</h1>
                <p>Xác định bán kính cắt (Cutoff Distance) để đo mật độ.</p>
                <div class="viz-box" style="height:400px; margin-top:20px;">
                    <canvas class="scene-canvas" data-scene="dc-intro" width="1000" height="400"></canvas>
                </div>
            </section>

            <section>
                <h2 class="text-red">Bước 2 - Vấn đề: Tính Tổng Serial</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-goc">Version Gốc</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-spinner"></i> Vòng lặp đơn luồng (Serial) duyệt qua hàng triệu phần tử.</li>
                            <li><i class="fa-solid fa-bed"></i> 1 nhân làm việc, 15 nhân còn lại "ngủ đông".</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="serial-sum" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <section>
                <h2 class="text-green">Bước 2 - Giải pháp: Parallel Reduction</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-final">Final Version</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-bolt"></i> Sử dụng <code>#pragma omp reduction(+:mean_c)</code>.</li>
                            <li><i class="fa-solid fa-diagram-project"></i> Thuật toán tự động cộng dồn theo mô hình Cây Nhị Phân O(log T) siêu tốc.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="parallel-reduce" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <!-- ================= BƯỚC 3: MẬT ĐỘ ================= -->

            <section>
                <h1 class="text-blue">BƯỚC 3: TÍNH MẬT ĐỘ (DENSITY)</h1>
                <p>Xác định mật độ cục bộ của từng điểm trong bán kính d_c.</p>
                <div class="viz-box" style="height:400px; margin-top:20px;">
                    <canvas class="scene-canvas" data-scene="density-intro" width="1000" height="400"></canvas>
                </div>
            </section>

            <section>
                <h2 class="text-red">Bước 3 - Vấn đề: Race Condition</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-goc">Version Gốc / 1 / 2</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-triangle-exclamation"></i> T1 và T2 cùng lúc +1 vào Mật độ của cùng 1 điểm mục tiêu.</li>
                            <li><i class="fa-solid fa-fire"></i> Gây ra lỗi ghi đè, làm sai lệch hoàn toàn mật độ thực sự của cụm.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="race-condition" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <section>
                <h2 class="text-green">Bước 3 - Giải pháp: Local Accumulation</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-final">Final Version</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-shield"></i> Giải pháp: Mỗi luồng cập nhật trên vùng Local Array của riêng nó.</li>
                            <li><i class="fa-solid fa-object-group"></i> Không bao giờ đụng chạm vùng nhớ của luồng khác. An toàn tuyệt đối.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="local-acc" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <!-- ================= BƯỚC 4: TÌM TÂM ================= -->

            <section>
                <h1 class="text-blue">BƯỚC 4: TÌM TÂM CỤM</h1>
                <p>Sắp xếp điểm theo thứ tự mật độ giảm dần để chọn Tâm Cụm đỉnh cao nhất.</p>
                <div class="viz-box" style="height:400px; margin-top:20px;">
                    <canvas class="scene-canvas" data-scene="center-intro" width="1000" height="400"></canvas>
                </div>
            </section>

            <section>
                <h2 class="text-red">Bước 4 - Vấn đề: Selection Sort O(N²)</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-goc">Version Gốc</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-bug"></i> Hàm <code>core_sort_desc()</code> dùng Selection Sort cực chậm.</li>
                            <li><i class="fa-solid fa-hourglass"></i> Tốn 15 giây "đóng băng" phần mềm chỉ để sắp xếp mảng 70.000 phần tử.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="selection-sort" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <section>
                <h2 class="text-green">Bước 4 - Giải pháp: Struct + Qsort (Final)</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-final">Final Version</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-box"></i> Đóng gói Value và Index vào <code>struct ValIdx</code> của C++.</li>
                            <li><i class="fa-solid fa-forward-fast"></i> Chạy <code>qsort()</code> chuẩn xác. Thời gian giảm từ 15s xuống <b>< 0.1s</b>.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="qsort-centers" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <!-- ================= BƯỚC 5: BFS ================= -->

            <section>
                <h1 class="text-blue">BƯỚC 5: XÂY CỤM BAN ĐẦU</h1>
                <p>Từ các Tâm Cụm, liên tục mở rộng cụm thông qua hàng đợi (hoạt động như thuật toán BFS).</p>
                <div class="viz-box" style="height:400px; margin-top:20px;">
                    <canvas class="scene-canvas" data-scene="bfs-intro" width="1000" height="400"></canvas>
                </div>
            </section>

            <section>
                <h2 class="text-red">Bước 5 - Vấn đề: Quét Tâm O(N²) (Version Gốc)</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-goc">Version Gốc</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-search"></i> Quá trình mở rộng cụm liên tục nạp điểm mới.</li>
                            <li><i class="fa-solid fa-bomb"></i> Hàm gốc quét lại TOÀN BỘ N điểm để tính lại trung bình cộng mỗi khi nạp thêm 1 điểm! Thiệt hại O(N² × d).</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="bfs-n2" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <section>
                <h2 class="text-green">Bước 5 - Giải pháp: Incremental Update (Version 1)</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-1">Version 1</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-calculator"></i> Lưu đệm bộ biến <code>csum</code> và <code>cnt</code>.</li>
                            <li><i class="fa-solid fa-plus"></i> Điểm mới nạp vào: <code>csum += X</code>. Tốc độ chuyển từ O(N) sang Phép cộng O(1)!</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="bfs-inc" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <section>
                <h2 class="text-yellow">Bước 5 - Lỗi Thuật Toán: Sai lệch ARI (Version 2)</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-2">Version 2</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-scale-unbalanced"></i> Version 2 ép quá trình xây cụm chạy Đa luồng.</li>
                            <li><i class="fa-solid fa-xmark"></i> Các Thread cướp ranh giới của nhau ngẫu nhiên. Mất hoàn toàn tính Deterministic. ARI tụt từ 0.407 xuống 0.376.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="bfs-race" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <section>
                <h2 class="text-green">Bước 5 - Giải pháp: Phục Hồi Cầu Nối (Final)</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-final">Final Version</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-rotate-left"></i> Chấp nhận chạy Đơn luồng (Serial) sử dụng Hàng đợi Queue (cơ chế BFS) để giữ đúng nguyên tắc phát triển ranh giới.</li>
                            <li><i class="fa-solid fa-circle-nodes"></i> Đưa (Enqueue) TẤT CẢ các láng giềng thỏa mãn vào Hàng đợi. Giải cứu <b>Bridge Node</b> để màu sắc tràn sâu vào viền cụm.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="bfs-bridge" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <!-- ================= BƯỚC 6: MATRIX A ================= -->

            <section>
                <h1 class="text-blue">BƯỚC 6: MA TRẬN LIÊN KẾT A</h1>
                <p>Tìm các cặp điểm ranh giới để bầu chọn nhãn.</p>
                <div class="viz-box" style="height:400px; margin-top:20px;">
                    <canvas class="scene-canvas" data-scene="matrix-intro" width="1000" height="400"></canvas>
                </div>
            </section>

            <section>
                <h2 class="text-red">Bước 6 - Vấn đề: Quét Ma Trận O(N²)</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-goc">Version Gốc / 1 / 2</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-table"></i> Khi điểm P có nhãn, thuật toán lặp qua N điểm để xem ai chứa P trong kNN.</li>
                            <li><i class="fa-solid fa-bomb"></i> Vòng lặp O(N² × K) bào mòn CPU liên tục vô ích.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="matrix-scan" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <section>
                <h2 class="text-green">Bước 6 - Giải pháp: Reverse-kNN (Final)</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-final">Final Version</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-crosshairs"></i> Xây dựng mảng Index <code>rknn[P]</code> chứa sẵn danh sách người trỏ đến P.</li>
                            <li><i class="fa-solid fa-bolt"></i> Bắn "Tia Laser" thẳng tới mục tiêu, O(N × K). Tốc độ cực hạn.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="matrix-rknn" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <!-- ================= BƯỚC 7: BẦU CHỌN ================= -->

            <section>
                <h1 class="text-blue">BƯỚC 7: BẦU CHỌN RANH GIỚI</h1>
                <p>Quyết định nhãn cho những điểm lấp lửng dựa vào số phiếu của láng giềng.</p>
                <div class="viz-box" style="height:400px; margin-top:20px;">
                    <canvas class="scene-canvas" data-scene="vote-intro" width="1000" height="400"></canvas>
                </div>
            </section>

            <section>
                <h2 class="text-red">Bước 7 - Vấn đề: Lỗi CPU Cache Thrashing</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-1">Version 1 / 2</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-microchip"></i> <b>False Sharing:</b> CPU tải dữ liệu theo Cache Line (64 Bytes).</li>
                            <li><i class="fa-solid fa-skull"></i> Biến đếm của T1 và T2 nằm sát nhau. T1 ghi đè sẽ làm hỏng (Invalidate) Cache của T2 liên tục.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="cache-thrash" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <section>
                <h2 class="text-green">Bước 7 - Giải pháp: Cache Padding 64B (Final)</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-final">Final Version</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-crop-simple"></i> Bơm biến rác (Padding) để ép vùng đếm của mỗi luồng to ĐÚNG 64 Bytes.</li>
                            <li><i class="fa-solid fa-unlock"></i> Luồng T1 và T2 sở hữu 2 Cache Line độc lập. Khai thác sức mạnh Đa nhân đạt mức 100% tuyến tính.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="cache-pad" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <!-- ================= BƯỚC 8: NGOẠI LAI ================= -->

            <section>
                <h1 class="text-blue">BƯỚC 8: GÁN NHÃN NGOẠI LAI</h1>
                <p>Loại bỏ các điểm nhiễu (Noise) để hoàn tất phân cụm thuật toán.</p>
                <div class="viz-box" style="height:400px; margin-top:20px;">
                    <canvas class="scene-canvas" data-scene="outlier-intro" width="1000" height="400"></canvas>
                </div>
            </section>

            <section>
                <h2 class="text-green">Bước 8 - Tối Ưu Hóa Tuyệt Đối (Final)</h2>
                <div class="slide-layout">
                    <div class="glass-panel">
                        <span class="v-tag v-final">Final Version</span>
                        <ul class="custom-list">
                            <li><i class="fa-solid fa-check-to-slot"></i> Cơ chế Cache Padding 64 Bytes tiếp tục được áp dụng cho việc gán ngoại lai.</li>
                            <li><i class="fa-solid fa-jet-fighter"></i> Tối đa hóa toàn bộ băng thông bộ nhớ RAM.</li>
                        </ul>
                    </div>
                    <div class="viz-box">
                        <canvas class="scene-canvas" data-scene="outlier-pad" width="800" height="600"></canvas>
                    </div>
                </div>
            </section>

            <!-- ================= KẾT LUẬN ================= -->

            <section>
                <h2 class="text-purple">Trực Quan Hóa Thực Tế (Fashion MNIST)</h2>
                <div class="glass-panel" style="display:flex; justify-content:center;">
                    <img src="output/plots/16threads_fashion_v1.png" style="height: 500px; border-radius: 12px; box-shadow: 0 0 40px rgba(181,55,242,0.6);">
                </div>
            </section>

            <section>
                <div style="margin-bottom: 20px; text-align: center;">
                    <i class="fa-solid fa-award text-green" style="font-size: 6em;"></i>
                </div>
                <h1 style="text-align: center; font-size:4em;" class="text-green">HOÀN TẤT</h1>
                <div class="glass-panel" style="text-align:center;">
                    <h3 class="text-blue" style="font-size:1.5em;">Siêu Kiến Trúc Đa Luồng</h3>
                    <p style="font-size:1.2em;">Giải quyết triệt để OOM RAM, loại trừ thắt cổ chai hệ điều hành và khai phóng hoàn toàn phần cứng CPU. Kiến trúc nhanh, bền vững và chính xác tuyệt đối.</p>
                </div>
            </section>

        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/reveal.min.js"></script>
    <script>
        Reveal.initialize({ 
            width: 1200,
            height: 700,
            margin: 0.1,
            hash: true, 
            transition: 'slide', 
            backgroundTransition: 'fade',
            controls: true,
            progress: true,
            center: true
        });

        // ================= CANVAS ENGINE =================
        const scenes = {};
        
        // Helper: Draw Node
        function drawNode(ctx, x, y, r, color, glow=false) {
            ctx.beginPath();
            ctx.arc(x, y, r, 0, Math.PI*2);
            ctx.fillStyle = color;
            if(glow) {
                ctx.shadowBlur = 20; ctx.shadowColor = color;
            }
            ctx.fill();
            ctx.shadowBlur = 0;
        }
        function drawText(ctx, txt, x, y, size, color, align="center") {
            ctx.font = `bold ${size}px 'Fira Code'`;
            ctx.fillStyle = color;
            ctx.textAlign = align;
            ctx.textBaseline = "middle";
            ctx.fillText(txt, x, y);
        }
        function drawLine(ctx, x1, y1, x2, y2, color, width=2) {
            ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
            ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
        }

        // Master Sim
        let ms_pts = [];
        let c1={x:200,y:200}, c2={x:600,y:300}, c3={x:400,y:500};
        for(let i=0; i<150; i++){
            let cen = i<50 ? c1 : (i<100 ? c2 : c3);
            if(Math.random()<0.1) ms_pts.push({x:Math.random()*800, y:Math.random()*600, r:3, c:'#333'});
            else ms_pts.push({x:cen.x+(Math.random()-0.5)*200, y:cen.y+(Math.random()-0.5)*200, r:3, c:'#555'});
        }
        scenes['master-sim'] = function(ctx, w, h, t) {
            let phase = (t % 12000) / 12000; 
            ctx.clearRect(0,0,w,h);
            
            ms_pts.forEach((p, i) => {
                let color = p.c; let r = p.r; let glow = false;
                if(phase > 0.1 && phase < 0.3) {
                    if(i%5===0) { color = '#00f3ff'; drawLine(ctx, p.x, p.y, p.x+20, p.y+20, 'rgba(0,243,255,0.3)'); }
                }
                if(phase > 0.3) {
                    if(i===0 || i===50 || i===100) { r=10; glow=true; color= i===0?'#00f3ff':(i===50?'#ff3366':'#00ff66'); }
                }
                if(phase > 0.5) {
                    if(i<50) color='#00f3ff'; else if(i<100) color='#ff3366'; else if(i<140) color='#00ff66';
                }
                if(phase > 0.8 && i>=140) color = '#111';
                drawNode(ctx, p.x, p.y, r, color, glow);
            });
            let txt = phase<0.1 ? "Khởi tạo..." : (phase<0.3 ? "Bước 1: kNN" : (phase<0.5 ? "Bước 2-4: Mật độ & Tâm" : (phase<0.8 ? "Bước 5: Xây cụm" : "Bước 8: Ngoại lai")));
            drawText(ctx, txt, 20, 30, 20, '#00ff66', 'left');
        };

        // Goals
        scenes['goals'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            let p = (Math.sin(t/500)+1)/2;
            let ramH = 400 - p*300; let cpuH = 100 + p*300; let ariH = 400;
            ctx.fillStyle='#333'; ctx.fillRect(150, 500, 500, 4);
            ctx.fillStyle='#ff3366'; ctx.fillRect(200, 500-ramH, 80, ramH); drawText(ctx, "RAM", 240, 530, 24, '#ff3366');
            ctx.fillStyle='#00f3ff'; ctx.fillRect(360, 500-cpuH, 80, cpuH); drawText(ctx, "CPU", 400, 530, 24, '#00f3ff');
            ctx.fillStyle='#00ff66'; ctx.fillRect(520, 500-ariH, 80, ariH); drawText(ctx, "ARI", 560, 530, 24, '#00ff66');
        };

        // kNN Intro
        let ki_pts = []; for(let i=0;i<40;i++) ki_pts.push({x:Math.random()*1000, y:Math.random()*400});
        scenes['knn-intro'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            let cx = w/2, cy = h/2;
            let radarR = (t/5) % 600;
            ctx.beginPath(); ctx.arc(cx,cy, radarR, 0, Math.PI*2); ctx.strokeStyle='rgba(0,243,255,0.2)'; ctx.lineWidth=5; ctx.stroke();
            
            ki_pts.forEach(p => {
                let d = Math.hypot(p.x-cx, p.y-cy);
                drawNode(ctx, p.x, p.y, 4, '#555');
                if(d < 150) {
                    drawLine(ctx, cx, cy, p.x, p.y, 'rgba(0,243,255,0.5)', 2);
                    drawNode(ctx, p.x, p.y, 6, '#00f3ff');
                    drawText(ctx, Math.floor(d), p.x+15, p.y, 14, '#00f3ff');
                }
            });
            drawNode(ctx, cx, cy, 10, '#00f3ff', true);
            drawText(ctx, "K = 5", cx, cy-25, 20, '#fff');
        };

        // RAM OOM
        scenes['ram-oom'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            let p = (t%3000)/3000;
            ctx.strokeStyle='#fff'; ctx.lineWidth=6; ctx.strokeRect(300, 100, 200, 400);
            let fillH = p*400;
            ctx.fillStyle = p > 0.9 ? '#ff3366' : (p > 0.6 ? '#ffde00' : '#00ff66');
            ctx.fillRect(303, 500-fillH, 194, fillH);
            for(let i=0; i<10; i++) {
                let dropY = ((t/2 + i*40) % 500);
                drawText(ctx, "[D_ij]", 400, dropY, 16, '#000');
            }
            if(p > 0.9) {
                let shake = (Math.random()-0.5)*10;
                drawText(ctx, "OOM CRASH!", 400+shake, 300+shake, 50, '#fff', "center");
                ctx.shadowBlur=30; ctx.shadowColor='#ff3366';
            }
        };

        // Mem Wall
        scenes['mem-wall'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            ctx.fillStyle='#00f3ff'; ctx.fillRect(100, 200, 150, 150); drawText(ctx, "CPU", 175, 250, 40, '#000');
            drawText(ctx, "0.5 ns", 175, 300, 20, '#000');
            
            ctx.fillStyle='#b537f2'; ctx.fillRect(550, 100, 150, 350); drawText(ctx, "RAM", 625, 250, 40, '#fff');
            drawText(ctx, "100 ns", 625, 300, 20, '#fff');

            ctx.fillStyle='#333'; ctx.fillRect(250, 260, 300, 30);
            let packX = 550 - ((t/5)%300);
            ctx.fillStyle='#ff3366'; ctx.fillRect(packX, 260, 30, 30);

            let wait = Math.floor(t/100)%4;
            drawText(ctx, "Waiting" + ".".repeat(wait), 175, 380, 24, '#ff3366');
        };

        // On The Fly
        scenes['on-the-fly'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            ctx.fillStyle='#333'; ctx.fillRect(100, 300, 400, 20); // Belt
            ctx.strokeStyle='#00f3ff'; ctx.lineWidth=4; ctx.strokeRect(550, 200, 150, 150); drawText(ctx, "K-Buffer", 625, 275, 24, '#00f3ff');
            ctx.fillStyle='#222'; ctx.fillRect(350, 400, 100, 100); drawText(ctx, "TRASH", 400, 450, 24, '#ff3366');

            let itemX = 100 + ((t/3)%500);
            let isTrash = Math.floor(t/1500)%2 === 0;
            let y = 270;
            if(isTrash && itemX > 400) { y += (itemX-400)*2; itemX=400; }
            if(!isTrash && itemX > 600) itemX = 600;

            ctx.fillStyle = isTrash ? '#ff3366' : '#00ff66';
            if(y < 500) {
                ctx.fillRect(itemX, y, 30, 30);
                drawText(ctx, isTrash?"99.9":"0.1", itemX+15, y+15, 12, '#000');
            }
        };

        // Lock Contention
        scenes['lock-contention'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            drawNode(ctx, 400, 300, 80, '#333', false);
            ctx.strokeStyle='#ff3366'; ctx.lineWidth=10; ctx.strokeRect(360, 260, 80, 80);
            drawText(ctx, "OS LOCK", 400, 300, 20, '#fff');

            for(let i=0; i<5; i++) {
                let ang = (i*Math.PI*2/5) + t/1000;
                let r = 200 - Math.abs(Math.sin(t/300 + i)*100);
                let tx = 400 + Math.cos(ang)*r;
                let ty = 300 + Math.sin(ang)*r;
                let color = r < 120 ? '#ff3366' : '#00f3ff';
                drawNode(ctx, tx, ty, 20, color);
                if(r < 120) drawText(ctx, "DENIED", tx, ty-30, 16, '#ff3366');
            }
        };

        // Pre Alloc
        scenes['pre-alloc'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            for(let i=0; i<4; i++) {
                let y = 100 + i*120;
                ctx.strokeStyle='#333'; ctx.lineWidth=4; ctx.beginPath(); ctx.moveTo(100,y); ctx.lineTo(600,y); ctx.stroke();
                ctx.strokeStyle='#fff'; ctx.strokeRect(600, y-30, 100, 60); drawText(ctx, "Buf "+i, 650, y, 20, '#fff');
                let tx = 100 + ((t/2 + i*200)%500);
                drawNode(ctx, tx, y, 20, '#00ff66', true);
            }
        };

        // Qsort
        scenes['qsort-nlogn'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            let arr = [9,5,2,7,1,8,3,6,4];
            let i_idx = Math.floor(t/1000)%9;
            let j_idx = Math.floor(t/100)%9;
            for(let k=0; k<9; k++) {
                let x = 100 + k*70;
                ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.strokeRect(x, 250, 60, 60);
                drawText(ctx, arr[k], x+30, 280, 30, '#fff');
                if(k === i_idx) { ctx.fillStyle='rgba(0,243,255,0.5)'; ctx.fillRect(x,250,60,60); drawText(ctx, "i", x+30, 340, 24, '#00f3ff'); }
                if(k === j_idx) { ctx.fillStyle='rgba(255,51,102,0.5)'; ctx.fillRect(x,250,60,60); drawText(ctx, "j", x+30, 220, 24, '#ff3366'); }
            }
            drawText(ctx, "O(N²)", 400, 450, 60, '#ff3366');
        };

        // Max Heap
        scenes['max-heap'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            drawLine(ctx, 400, 150, 250, 300, '#fff'); drawLine(ctx, 400, 150, 550, 300, '#fff');
            drawNode(ctx, 250, 300, 40, '#333'); drawText(ctx, "10", 250, 300, 30, '#fff');
            drawNode(ctx, 550, 300, 40, '#333'); drawText(ctx, "15", 550, 300, 30, '#fff');
            
            let phase = (t%3000)/3000;
            if(phase < 0.5) {
                drawNode(ctx, 400, 150, 50, '#ff3366', true); drawText(ctx, "42", 400, 150, 40, '#fff');
                let dropY = -50 + phase*2*200;
                drawNode(ctx, 400, dropY, 40, '#00ff66', true); drawText(ctx, "8", 400, dropY, 30, '#000');
            } else {
                let flyX = 400 - (phase-0.5)*2*400; let flyY = 150 + (phase-0.5)*2*400;
                drawNode(ctx, flyX, flyY, 50, '#ff3366'); drawText(ctx, "42", flyX, flyY, 40, '#fff');
                drawNode(ctx, 400, 150, 50, '#00ff66', true); drawText(ctx, "8", 400, 150, 40, '#000');
            }
        };

        // Early Exit
        scenes['early-exit'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            drawText(ctx, "dist = x² + y² + z²", 400, 200, 50, '#fff');
            let phase = (t%2000)/2000;
            if(phase > 0.3) {
                drawText(ctx, "x² + y² > MAX", 400, 300, 60, '#ff3366');
            }
            if(phase > 0.6) {
                ctx.strokeStyle='#ff3366'; ctx.lineWidth=10;
                ctx.beginPath(); ctx.moveTo(500, 150); ctx.lineTo(650, 250); ctx.stroke();
                drawText(ctx, "BREAK!", 650, 150, 50, '#00ff66');
            }
        };

        // dc-intro
        scenes['dc-intro'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            for(let i=0; i<100; i++) drawNode(ctx, 100+(i*17)%800, 50+(i*23)%300, 3, '#444');
            drawNode(ctx, 500, 200, 8, '#fff');
            let r = (t/10)%200;
            ctx.beginPath(); ctx.arc(500, 200, r, 0, Math.PI*2); ctx.strokeStyle='#ffde00'; ctx.lineWidth=3; ctx.setLineDash([10,10]); ctx.stroke(); ctx.setLineDash([]);
            if(r > 150) { drawText(ctx, "d_c = 2.0%", 500, 370, 30, '#00ff66'); }
        };

        // Serial Sum
        scenes['serial-sum'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            for(let i=0; i<15; i++) {
                ctx.strokeStyle='#333'; ctx.strokeRect(50+i*45, 300, 40, 40);
            }
            let idx = Math.floor(t/200)%15;
            ctx.fillStyle='#ff3366'; ctx.fillRect(50+idx*45, 300, 40, 40);
            drawText(ctx, "Core 0: " + idx, 100, 250, 24, '#ff3366');
            drawText(ctx, "Cores 1-15: Zzz...", 400, 250, 24, '#555');
        };

        // Parallel Reduce
        scenes['parallel-reduce'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            let phase = Math.floor(t/1000)%3;
            for(let i=0; i<8; i++) drawNode(ctx, 100+i*80, 400, 20, phase===0?'#00f3ff':'#333');
            for(let i=0; i<4; i++) {
                drawLine(ctx, 100+i*160, 400, 140+i*160, 300, '#555');
                drawLine(ctx, 180+i*160, 400, 140+i*160, 300, '#555');
                drawNode(ctx, 140+i*160, 300, 25, phase===1?'#00f3ff':'#333');
            }
            for(let i=0; i<2; i++) {
                drawLine(ctx, 140+i*320, 300, 220+i*320, 200, '#555');
                drawLine(ctx, 300+i*320, 300, 220+i*320, 200, '#555');
                drawNode(ctx, 220+i*320, 200, 30, phase===2?'#00f3ff':'#333');
            }
            drawLine(ctx, 220, 200, 380, 100, '#555');
            drawLine(ctx, 540, 200, 380, 100, '#555');
            drawNode(ctx, 380, 100, 40, '#00ff66', true); drawText(ctx, "SUM", 380, 100, 20, '#000');
        };

        // Density Intro
        scenes['density-intro'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            drawNode(ctx, 500, 200, 15, '#ffde00', true);
            ctx.beginPath(); ctx.arc(500, 200, 150, 0, Math.PI*2); ctx.strokeStyle='rgba(255,222,0,0.3)'; ctx.stroke();
            let count = 0;
            for(let i=0; i<20; i++) {
                let px = 500 + Math.cos(i)*120; let py = 200 + Math.sin(i*2)*120;
                if(Math.sin(t/300 + i) > 0) { drawNode(ctx, px, py, 6, '#00f3ff'); drawLine(ctx, 500, 200, px, py, 'rgba(0,243,255,0.5)'); count++; }
                else drawNode(ctx, px, py, 4, '#555');
            }
            drawText(ctx, "Density rho = " + count, 500, 380, 30, '#00ff66');
        };

        // Race Condition
        scenes['race-condition'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            drawNode(ctx, 200, 200, 30, '#00f3ff'); drawText(ctx, "T1", 200, 200, 20, '#000');
            drawNode(ctx, 600, 200, 30, '#ffde00'); drawText(ctx, "T2", 600, 200, 20, '#000');
            
            let phase = (t%2000)/2000;
            if(phase < 0.5) {
                drawLine(ctx, 200, 200, 200+phase*400, 200+phase*400, '#00f3ff', 5);
                drawLine(ctx, 600, 200, 600-phase*400, 200+phase*400, '#ffde00', 5);
                drawNode(ctx, 400, 400, 40, '#333');
            } else {
                drawNode(ctx, 400, 400, 60, '#ff3366', true);
                drawText(ctx, "RACE ERROR!", 400, 500, 40, '#ff3366');
            }
        };

        // Local Acc
        scenes['local-acc'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            ctx.strokeStyle='#00f3ff'; ctx.strokeRect(150, 150, 100, 100); drawText(ctx, "Local T1", 200, 200, 20, '#00f3ff');
            ctx.strokeStyle='#ffde00'; ctx.strokeRect(550, 150, 100, 100); drawText(ctx, "Local T2", 600, 200, 20, '#ffde00');
            let phase = (t%2000)/2000;
            if(phase > 0.5) {
                drawLine(ctx, 200, 250, 400, 400, '#00f3ff', 4);
                drawLine(ctx, 600, 250, 400, 400, '#ffde00', 4);
                ctx.fillStyle='#00ff66'; ctx.fillRect(350, 400, 100, 100); drawText(ctx, "SUM", 400, 450, 24, '#000');
            }
        };

        // Center Intro
        scenes['center-intro'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            for(let i=0; i<10; i++) {
                let hBar = 250 - i*20;
                ctx.fillStyle = i<3 ? '#b537f2' : '#00f3ff';
                ctx.fillRect(100+i*60, 400-hBar, 40, hBar);
                if(i<3) {
                    let cY = 400-hBar-30 + Math.sin(t/200)*10;
                    drawText(ctx, "👑", 120+i*60, cY, 40, '#fff');
                }
            }
        };

        // Selection Sort
        scenes['selection-sort'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            for(let i=0; i<15; i++) {
                ctx.fillStyle='#333'; ctx.fillRect(50+i*45, 300, 40, Math.sin(i*99)*50 + 80);
            }
            let idx = Math.floor(t/100)%15;
            ctx.fillStyle='#ff3366'; ctx.fillRect(50+idx*45, 300, 40, Math.sin(idx*99)*50 + 80);
            drawText(ctx, "O(N²)", 400, 200, 60, '#ff3366');
        };

        // Qsort Centers
        scenes['qsort-centers'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            let phase = (t%2000)/2000;
            ctx.fillStyle='#ffde00'; ctx.fillRect(350, 100, 100, 40); drawText(ctx, "PIVOT", 400, 120, 20, '#000');
            ctx.fillStyle='#00f3ff'; ctx.fillRect(150 - phase*50, 200, 200, 40); drawText(ctx, "< PIVOT", 250 - phase*50, 220, 20, '#000');
            ctx.fillStyle='#ff3366'; ctx.fillRect(450 + phase*50, 200, 200, 40); drawText(ctx, "> PIVOT", 550 + phase*50, 220, 20, '#000');
            drawText(ctx, "O(N log N)", 400, 400, 60, '#00ff66');
        };

        // BFS Intro
        let bf_pts = []; for(let i=0;i<30;i++) bf_pts.push({x:Math.random()*800, y:Math.random()*600, c:'#444'});
        bf_pts[0] = {x:400,y:300, c:'#ff3366'};
        scenes['bfs-intro'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            let rad = (t/5)%500;
            bf_pts.forEach((p, i) => {
                let d = Math.hypot(p.x-400, p.y-300);
                if(d < rad) { p.c = '#ff3366'; drawLine(ctx, 400, 300, p.x, p.y, 'rgba(255,51,102,0.3)'); }
                drawNode(ctx, p.x, p.y, i===0?15:6, p.c, i===0);
            });
        };

        // BFS N2
        scenes['bfs-n2'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            drawNode(ctx, 400, 300, 20, '#ff3366');
            let scanY = (t)%600;
            ctx.fillStyle='rgba(255,255,255,0.2)'; ctx.fillRect(0, scanY, 800, 20);
            for(let i=0; i<20; i++) drawNode(ctx, 100+i*30, 200+(i*77)%300, 5, '#555');
            drawText(ctx, "Quét toàn bộ N điểm", 400, 500, 40, '#ff3366');
        };

        // BFS Inc
        scenes['bfs-inc'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            drawNode(ctx, 400, 300, 40, '#00ff66', true); drawText(ctx, "C", 400, 300, 30, '#000');
            let px = 400 + Math.cos(t/300)*150; let py = 300 + Math.sin(t/300)*150;
            drawNode(ctx, px, py, 15, '#fff'); drawLine(ctx, 400, 300, px, py, '#fff');
            drawText(ctx, "csum += X", 400, 500, 50, '#00ff66');
        };

        // BFS Race
        scenes['bfs-race'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            drawNode(ctx, 200, 300, 30, '#00f3ff'); drawNode(ctx, 600, 300, 30, '#ff3366');
            let phase = Math.floor(t/100)%2;
            let c = phase===0 ? '#00f3ff' : '#ff3366';
            drawNode(ctx, 400, 300, 20, c, true);
            drawLine(ctx, 200, 300, 400, 300, '#00f3ff', 5); drawLine(ctx, 600, 300, 400, 300, '#ff3366', 5);
            drawText(ctx, "Tranh giành Ranh giới!", 400, 450, 40, '#ffde00');
        };

        // BFS Bridge
        scenes['bfs-bridge'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            drawNode(ctx, 200, 300, 40, '#00f3ff'); drawNode(ctx, 600, 300, 40, '#ff3366');
            drawNode(ctx, 400, 300, 20, '#00ff66', true); drawText(ctx, "Bridge", 400, 260, 20, '#00ff66');
            let offset = (t/5)%20;
            ctx.beginPath(); ctx.moveTo(200, 300); ctx.lineTo(400, 300); ctx.strokeStyle='#00f3ff'; ctx.lineWidth=5; ctx.setLineDash([10,10]); ctx.lineDashOffset = -offset; ctx.stroke();
            ctx.beginPath(); ctx.moveTo(400, 300); ctx.lineTo(600, 300); ctx.strokeStyle='#ff3366'; ctx.lineWidth=5; ctx.lineDashOffset = offset; ctx.stroke(); ctx.setLineDash([]);
        };

        // Matrix Intro
        scenes['matrix-intro'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            for(let i=0; i<5; i++) {
                for(let j=0; j<5; j++) {
                    ctx.strokeStyle='#333'; ctx.strokeRect(200+i*80, 100+j*80, 75, 75);
                }
            }
            let a = Math.floor(t/200)%5; let b = Math.floor(t/300)%5;
            ctx.fillStyle='#00f3ff'; ctx.fillRect(200+a*80, 100+b*80, 75, 75);
        };

        // Matrix Scan
        scenes['matrix-scan'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            drawNode(ctx, 400, 300, 20, '#b537f2');
            let scanR = (t)%600;
            ctx.beginPath(); ctx.arc(400, 300, scanR, 0, Math.PI*2); ctx.strokeStyle='rgba(255,51,102,0.5)'; ctx.lineWidth=20; ctx.stroke();
            drawText(ctx, "Quét N điểm", 400, 500, 40, '#ff3366');
        };

        // Matrix RKNN
        scenes['matrix-rknn'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            drawNode(ctx, 400, 300, 20, '#b537f2');
            let pts = [{x:200,y:100}, {x:600,y:200}, {x:300,y:500}];
            pts.forEach(p => {
                drawNode(ctx, p.x, p.y, 10, '#555');
                if((t%1000) > 500) { drawLine(ctx, 400, 300, p.x, p.y, '#b537f2', 4); drawNode(ctx, p.x, p.y, 10, '#00f3ff'); }
            });
            drawText(ctx, "O(N x K)", 400, 550, 50, '#00ff66');
        };

        // Vote Intro
        scenes['vote-intro'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            for(let i=0; i<5; i++) drawNode(ctx, 200+Math.cos(i)*50, 300+Math.sin(i)*50, 10, '#00f3ff');
            for(let i=0; i<3; i++) drawNode(ctx, 600+Math.cos(i)*50, 300+Math.sin(i)*50, 10, '#ff3366');
            let phase = Math.floor(t/500)%2;
            drawNode(ctx, 400, 300, 20, phase===0?'#fff':'#00f3ff', phase!==0);
            drawText(ctx, "5 vs 3", 400, 200, 40, '#fff');
        };

        // Cache Thrash
        scenes['cache-thrash'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            ctx.strokeStyle='#fff'; ctx.lineWidth=4; ctx.strokeRect(200, 250, 400, 100);
            ctx.beginPath(); ctx.moveTo(400, 250); ctx.lineTo(400, 350); ctx.stroke();
            drawText(ctx, "Red_Cnt", 300, 300, 20, '#fff'); drawText(ctx, "Blue_Cnt", 500, 300, 20, '#fff');
            drawNode(ctx, 300, 100, 30, '#ff3366'); drawText(ctx, "T1", 300, 100, 20, '#000');
            drawNode(ctx, 500, 500, 30, '#00f3ff'); drawText(ctx, "T2", 500, 500, 20, '#000');
            
            let phase = Math.floor(t/500)%2;
            if(phase===0) { drawLine(ctx, 300, 130, 300, 250, '#ff3366', 5); drawNode(ctx, 500, 300, 40, '#ffde00', true); }
            else { drawLine(ctx, 500, 470, 500, 350, '#00f3ff', 5); drawNode(ctx, 300, 300, 40, '#ffde00', true); }
            drawText(ctx, "Cache Invalidation!", 400, 150, 30, '#ffde00');
        };

        // Cache Pad
        scenes['cache-pad'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            ctx.strokeStyle='#fff'; ctx.lineWidth=4; ctx.strokeRect(100, 250, 250, 100); drawText(ctx, "Red (64B)", 225, 300, 20, '#fff');
            ctx.strokeStyle='#fff'; ctx.lineWidth=4; ctx.strokeRect(450, 250, 250, 100); drawText(ctx, "Blue (64B)", 575, 300, 20, '#fff');
            
            drawNode(ctx, 225, 100, 30, '#ff3366'); drawLine(ctx, 225, 130, 225, 250, '#ff3366', 5);
            drawNode(ctx, 575, 500, 30, '#00f3ff'); drawLine(ctx, 575, 470, 575, 350, '#00f3ff', 5);
            drawText(ctx, "Độc lập 100%", 400, 450, 40, '#00ff66');
        };

        // Outlier Intro
        scenes['outlier-intro'] = function(ctx, w, h, t) {
            ctx.clearRect(0,0,w,h);
            drawNode(ctx, 400, 300, 20, '#00f3ff', true);
            let phase = Math.floor(t/1000)%2;
            if(phase===1) { drawNode(ctx, 600, 400, 15, '#222'); drawText(ctx, "X", 600, 400, 20, '#ff3366'); }
            else { drawNode(ctx, 600, 400, 15, '#fff'); }
            drawText(ctx, "Noise = -1", 400, 200, 40, '#fff');
        };

        // Outlier Pad
        scenes['outlier-pad'] = scenes['cache-pad'];

        // Core Render Loop
        const canvases = document.querySelectorAll('.scene-canvas');
        function render() {
            let now = Date.now();
            canvases.forEach(c => {
                // Check if visible (Reveal.js removes display:none but uses sections)
                if(c.offsetParent !== null) {
                    let scene = c.getAttribute('data-scene');
                    if(scenes[scene]) scenes[scene](c.getContext('2d'), c.width, c.height, now);
                }
            });
            requestAnimationFrame(render);
        }
        render();

    </script>
</body>
</html>
"""
with open('C:\\BTL_LTSS\\cpu_parallel\\dpc_aknn_optimization_presentation.html', 'w', encoding='utf-8') as f:
    f.write(html)
